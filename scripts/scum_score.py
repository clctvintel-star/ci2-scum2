import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from anthropic import Anthropic
from google import genai
from openai import OpenAI

from utils import load_configs, load_env, ensure_dir


# --------------------------------------------------
# args
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SCUM2 event-level scorer")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for testing")
    parser.add_argument(
        "--debug-primary-b",
        action="store_true",
        help="Print raw primary_b outputs for debugging",
    )
    return parser.parse_args()


# --------------------------------------------------
# helpers
# --------------------------------------------------

class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def latest_parquet(directory: str) -> Path:
    files = sorted(Path(directory).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    return files[-1]


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def autosave_df(df: pd.DataFrame, outdir: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outfile = Path(outdir) / f"scored_events_{ts}.parquet"
    df.to_parquet(outfile, index=False)
    return outfile


def scum_score(sentiment: Optional[float], confidence: Optional[float]) -> Optional[float]:
    if sentiment is None or confidence is None:
        return None
    try:
        return round(float(sentiment) * math.sqrt(float(confidence)), 4)
    except Exception:
        return None


def clamp_float(value: Optional[float], min_val: float, max_val: float) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return max(min_val, min(max_val, value))


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


# --------------------------------------------------
# parsing
# --------------------------------------------------

def parse_relevance_response(text: Any) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    if not isinstance(text, str):
        return None, None, None

    decision = None
    confidence = None
    reason = None

    m = re.search(r"Decision:\s*(KEEP|DROP|UNCERTAIN)", text, flags=re.IGNORECASE)
    if m:
        decision = m.group(1).upper()

    m = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if m:
        try:
            confidence = float(m.group(1))
        except Exception:
            confidence = None

    m = re.search(r"Reason:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()

    confidence = clamp_float(confidence, 0.0, 1.0)
    return decision, confidence, reason


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def coerce_incomplete_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Conservative repair for truncated JSON like:
    {"sentiment": 0.2, "confidence": 0.7, "explanation": "..."
    """
    if not isinstance(text, str):
        return None

    if "sentiment" not in text:
        return None

    sent_match = re.search(r'"sentiment"\s*:\s*(null|-?\d+(?:\.\d+)?)', text)
    conf_match = re.search(r'"confidence"\s*:\s*(null|\d+(?:\.\d+)?)', text)
    expl_match = re.search(r'"explanation"\s*:\s*"([^"]*)', text)

    if not sent_match or not conf_match:
        return None

    payload: Dict[str, Any] = {}

    s = sent_match.group(1)
    payload["sentiment"] = None if s == "null" else float(s)

    c = conf_match.group(1)
    payload["confidence"] = None if c == "null" else float(c)

    payload["explanation"] = expl_match.group(1).strip() if expl_match else None
    return payload


def extract_json_payload(text: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None

    text = strip_code_fences(text)
    if not text:
        return None

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        block = m.group(0)

        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

        try:
            block2 = re.sub(r"(?<!\\)'", '"', block)
            payload = json.loads(block2)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    repaired = coerce_incomplete_json(text)
    if isinstance(repaired, dict):
        return repaired

    return None


def parse_sentiment_response(text: Any) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    payload = extract_json_payload(text)
    if not payload:
        return None, None, None

    sentiment = payload.get("sentiment")
    confidence = payload.get("confidence")
    explanation = payload.get("explanation") or payload.get("reason") or payload.get("rationale")

    try:
        if sentiment is not None:
            sentiment = float(sentiment)
    except Exception:
        sentiment = None

    try:
        if confidence is not None:
            confidence = float(confidence)
    except Exception:
        confidence = None

    sentiment = clamp_float(sentiment, -1.0, 1.0)
    confidence = clamp_float(confidence, 0.0, 1.0)

    if sentiment is None:
        confidence = 0.0 if confidence is None else confidence

    if isinstance(explanation, str):
        explanation = explanation.strip()
    else:
        explanation = None

    return sentiment, confidence, explanation


def is_missing_score(sentiment: Optional[float], confidence: Optional[float]) -> bool:
    return sentiment is None or confidence is None


def should_trigger_solomon(
    a_s: Optional[float],
    a_c: Optional[float],
    a_scum: Optional[float],
    b_s: Optional[float],
    b_c: Optional[float],
    b_scum: Optional[float],
    thresh: float,
) -> bool:
    if is_missing_score(a_s, a_c) or is_missing_score(b_s, b_c):
        return True

    if a_s == 0 and b_s == 0 and a_c == 0 and b_c == 0:
        return True

    if a_s * b_s < 0:
        return True

    if a_scum is None or b_scum is None:
        return True

    if abs(a_scum - b_scum) > thresh:
        return True

    return False


# --------------------------------------------------
# parent context
# --------------------------------------------------

def normalize_reddit_id(raw_id: Any) -> str:
    raw_id = safe_text(raw_id)
    if not raw_id:
        return ""
    return re.sub(r"^(t1_|t3_)", "", raw_id)


def build_parent_lookup(events_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      post_text_map[post_id] = post event_text
      comment_text_map[comment_id] = comment event_text
    """
    post_text_map: Dict[str, str] = {}
    comment_text_map: Dict[str, str] = {}

    for _, row in events_df.iterrows():
        event_text = safe_text(row.get("event_text"))
        post_id = normalize_reddit_id(row.get("post_id"))
        comment_id = normalize_reddit_id(row.get("comment_id"))
        row_type = safe_text(row.get("type")).lower()

        if row_type == "post":
            if post_id and event_text and post_id not in post_text_map:
                post_text_map[post_id] = event_text

        if comment_id and event_text and comment_id not in comment_text_map:
            comment_text_map[comment_id] = event_text

    return post_text_map, comment_text_map


def get_parent_text(
    row: pd.Series,
    post_text_map: Dict[str, str],
    comment_text_map: Dict[str, str],
) -> str:
    parent_id = safe_text(row.get("parent_id"))
    if not parent_id:
        return ""

    if parent_id.startswith("t1_"):
        parent_comment_id = normalize_reddit_id(parent_id)
        return comment_text_map.get(parent_comment_id, "")

    if parent_id.startswith("t3_"):
        parent_post_id = normalize_reddit_id(parent_id)
        return post_text_map.get(parent_post_id, "")

    # fallback if raw parent_id has no prefix
    parent_norm = normalize_reddit_id(parent_id)
    if parent_norm in comment_text_map:
        return comment_text_map[parent_norm]
    if parent_norm in post_text_map:
        return post_text_map[parent_norm]

    return ""


# --------------------------------------------------
# clients
# --------------------------------------------------

def build_clients(env):
    anthropic_client = Anthropic(api_key=env["ANTHROPIC_API_KEY"])
    openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])
    gemini_client = genai.Client(api_key=env["GOOGLE_API_KEY"])
    return anthropic_client, openai_client, gemini_client


# --------------------------------------------------
# model calls
# --------------------------------------------------

def call_anthropic(
    client,
    model_name: str,
    prompt: str,
    timeout_seconds: int = 20,
    max_tokens: int = 300,
    temperature: float = 0.1,
    retry_limit: int = 3,
) -> str:
    last_err = None

    for _ in range(retry_limit):
        try:
            resp = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system="Return only the requested format. Output must be complete, valid, and final.",
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_seconds,
            )
            return resp.content[0].text.strip()
        except Exception as e:
            last_err = e
            print(f"Anthropic error ({model_name}), retrying: {e}")
            time.sleep(1)

    raise RuntimeError(f"Anthropic failed for {model_name}: {last_err}")


def call_openai(
    client,
    model_name: str,
    prompt: str,
    timeout_seconds: int = 20,
    max_tokens: int = 300,
    temperature: float = 0.1,
    retry_limit: int = 3,
) -> str:
    last_err = None

    for _ in range(retry_limit):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only the requested format. Output must be complete, valid, and final.",
                    },
                    {"role": "user", "content": prompt},
                ],
                timeout=timeout_seconds,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            print(f"OpenAI error ({model_name}), retrying: {e}")
            time.sleep(1)

    raise RuntimeError(f"OpenAI failed for {model_name}: {last_err}")


def call_gemini(
    client,
    model_name: str,
    prompt: str,
    timeout_seconds: int = 20,
    max_output_tokens: int = 220,
    temperature: float = 0.1,
    retry_limit: int = 3,
) -> str:
    last_err = None

    for _ in range(retry_limit):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "response_mime_type": "application/json",
                },
            )
            text = getattr(resp, "text", None)
            if text:
                return text.strip()
            return str(resp)
        except Exception as e:
            last_err = e
            print(f"Gemini error ({model_name}), retrying: {e}")
            time.sleep(1)

    raise RuntimeError(f"Gemini failed for {model_name}: {last_err}")


def call_model(
    model_name: str,
    prompt: str,
    anthropic_client,
    openai_client,
    gemini_client,
    timeout_seconds: int = 20,
    max_tokens: int = 300,
    temperature: float = 0.1,
    retry_limit: int = 3,
) -> str:
    if model_name.startswith("claude"):
        return call_anthropic(
            client=anthropic_client,
            model_name=model_name,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_limit=retry_limit,
        )

    if model_name.startswith("gpt"):
        return call_openai(
            client=openai_client,
            model_name=model_name,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_limit=retry_limit,
        )

    if model_name.startswith("gemini"):
        return call_gemini(
            client=gemini_client,
            model_name=model_name,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_tokens,
            temperature=temperature,
            retry_limit=retry_limit,
        )

    raise ValueError(f"Unsupported model family: {model_name}")


def call_and_parse_sentiment(
    model_name: str,
    prompt: str,
    anthropic_client,
    openai_client,
    gemini_client,
    timeout_seconds: int,
    max_tokens: int,
    temperature: float,
    transport_retry_limit: int,
    semantic_retry_limit: int = 2,
) -> Tuple[Optional[float], Optional[float], Optional[str], str]:
    """
    Makes up to semantic_retry_limit attempts to get parseable JSON.
    Transport retries happen inside call_model.
    """
    last_raw = ""

    for attempt in range(semantic_retry_limit):
        effective_prompt = prompt
        if attempt > 0:
            effective_prompt = (
                prompt
                + "\n\nIMPORTANT: Your previous output was invalid or incomplete. "
                  "Return one complete valid JSON object only with keys "
                  '"sentiment", "confidence", and "explanation". '
                  "No markdown fences. No extra text."
            )

        raw = call_model(
            model_name=model_name,
            prompt=effective_prompt,
            anthropic_client=anthropic_client,
            openai_client=openai_client,
            gemini_client=gemini_client,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_limit=transport_retry_limit,
        )
        last_raw = raw

        sentiment, confidence, explanation = parse_sentiment_response(raw)
        if confidence is not None:
            return sentiment, confidence, explanation, raw

    return None, None, None, last_raw


# --------------------------------------------------
# prompt payload
# --------------------------------------------------

def build_format_payload(
    row,
    parent_text: str = "",
    model_a_sentiment=None,
    model_a_confidence=None,
    model_b_sentiment=None,
    model_b_confidence=None,
):
    thread_title = safe_text(row.get("thread_title"))
    thread_text = safe_text(row.get("thread_text"))
    event_text = safe_text(row.get("event_text"))
    matched_aliases = safe_text(row.get("matched_aliases"))
    candidate_source = safe_text(row.get("candidate_source"))
    firm_name = safe_text(row.get("canonical_name"))
    parent_text = safe_text(parent_text)

    return SafeDict(
        {
            "FIRM_NAME": firm_name,
            "FUND_NAME": firm_name,
            "MATCHED_ALIASES": matched_aliases,
            "CANDIDATE_SOURCE": candidate_source,
            "THREAD_TITLE": thread_title,
            "THREAD_TEXT": thread_text,
            "THREAD_POST": thread_text,
            "PARENT_TEXT": parent_text,
            "EVENT_TEXT": event_text,
            "MODEL_A_SENTIMENT": "" if model_a_sentiment is None else model_a_sentiment,
            "MODEL_A_CONFIDENCE": "" if model_a_confidence is None else model_a_confidence,
            "MODEL_B_SENTIMENT": "" if model_b_sentiment is None else model_b_sentiment,
            "MODEL_B_CONFIDENCE": "" if model_b_confidence is None else model_b_confidence,
            # backward-compat placeholders
            "title": thread_title,
            "post": f"Thread context:\n{thread_text}\n\nParent text:\n{parent_text}\n\nEvent text:\n{event_text}",
            "comments": "",
        }
    )


# --------------------------------------------------
# main
# --------------------------------------------------

def main():
    args = parse_args()

    _, paths, settings = load_configs()
    env = load_env(paths["env_path"])

    ensure_dir(paths["scored_events_dir"])

    anthropic_client, openai_client, gemini_client = build_clients(env)

    relevance_prompt_template = read_text(Path("prompts/event_firm_relevance_prompt.txt"))
    uncertain_prompt_template = read_text(Path("prompts/event_firm_uncertain_prompt.txt"))
    primary_prompt_template = read_text(Path("prompts/scum_primary_prompt.txt"))
    tiebreaker_prompt_template = read_text(Path("prompts/scum_tiebreaker_prompt.txt"))

    relevance_model = settings["models"]["score"]["relevance_filter"]
    uncertain_model = settings["models"]["score"]["uncertain_adjudicator"]
    primary_a_model = settings["models"]["score"]["primary_a"]
    primary_b_model = settings["models"]["score"]["primary_b"]
    tiebreaker_model = settings["models"]["score"]["tiebreaker"]

    retry_limit = settings["scoring"]["retry_limit"]
    request_timeout = settings["scoring"]["request_timeout"]
    autosave_every = settings["scoring"]["autosave_every_n_rows"]
    disagree_threshold = settings["scoring"]["disagree_threshold"]
    uncertain_keep_threshold = settings["scoring"].get("uncertain_keep_threshold", 0.92)

    latest_event_file = latest_parquet(paths["event_firm_dir"])
    latest_thread_file = latest_parquet(paths["thread_firm_dir"])

    event_df = pd.read_parquet(latest_event_file)
    thread_df = pd.read_parquet(latest_thread_file)

    post_text_map, comment_text_map = build_parent_lookup(event_df)

    merge_cols = [
        "post_id",
        "canonical_name",
        "thread_title",
        "thread_text",
        "llm_rationale",
    ]

    scored_df = event_df.merge(
        thread_df[merge_cols].drop_duplicates(subset=["post_id", "canonical_name"]),
        on=["post_id", "canonical_name"],
        how="left",
    )

    scored_df["parent_text"] = scored_df.apply(
        lambda r: get_parent_text(r, post_text_map=post_text_map, comment_text_map=comment_text_map),
        axis=1,
    )

    if args.limit is not None:
        scored_df = scored_df.head(args.limit).copy()

    out_cols = [
        "parent_text",
        "relevance_decision",
        "relevance_confidence",
        "relevance_reason",
        "effective_relevance_decision",
        "uncertain_adjudication_decision",
        "uncertain_adjudication_confidence",
        "uncertain_adjudication_reason",
        "primary_a_sentiment",
        "primary_a_confidence",
        "primary_a_scum",
        "primary_b_sentiment",
        "primary_b_confidence",
        "primary_b_scum",
        "tiebreaker_sentiment",
        "tiebreaker_confidence",
        "tiebreaker_scum",
        "tiebreaker_reason",
        "final_sentiment",
        "final_confidence",
        "final_scum",
        "solomon_triggered",
    ]
    for c in out_cols:
        if c not in scored_df.columns:
            scored_df[c] = None

    rows_since_save = 0

    for idx, row in scored_df.iterrows():
        parent_text = safe_text(row.get("parent_text"))

        relevance_payload = build_format_payload(row, parent_text=parent_text)
        relevance_prompt = relevance_prompt_template.format_map(relevance_payload)

        relevance_text = call_model(
            model_name=relevance_model,
            prompt=relevance_prompt,
            anthropic_client=anthropic_client,
            openai_client=openai_client,
            gemini_client=gemini_client,
            timeout_seconds=request_timeout,
            max_tokens=160,
            temperature=0.0,
            retry_limit=retry_limit,
        )

        rel_decision, rel_conf, rel_reason = parse_relevance_response(relevance_text)

        scored_df.at[idx, "relevance_decision"] = rel_decision
        scored_df.at[idx, "relevance_confidence"] = rel_conf
        scored_df.at[idx, "relevance_reason"] = rel_reason

        effective_rel = rel_decision

        if rel_decision == "DROP":
            scored_df.at[idx, "effective_relevance_decision"] = "DROP"
            print(f"DROP: {row['event_id']} / {row['canonical_name']}")
            rows_since_save += 1

            if rows_since_save >= autosave_every:
                outfile = autosave_df(scored_df, paths["scored_events_dir"])
                print(f"AUTOSAVED: {outfile}")
                rows_since_save = 0

            continue

        if rel_decision == "UNCERTAIN":
            uncertain_payload = build_format_payload(row, parent_text=parent_text)
            uncertain_prompt = uncertain_prompt_template.format_map(uncertain_payload)

            uncertain_text = call_model(
                model_name=uncertain_model,
                prompt=uncertain_prompt,
                anthropic_client=anthropic_client,
                openai_client=openai_client,
                gemini_client=gemini_client,
                timeout_seconds=request_timeout,
                max_tokens=180,
                temperature=0.0,
                retry_limit=retry_limit,
            )

            u_decision, u_conf, u_reason = parse_relevance_response(uncertain_text)

            scored_df.at[idx, "uncertain_adjudication_decision"] = u_decision
            scored_df.at[idx, "uncertain_adjudication_confidence"] = u_conf
            scored_df.at[idx, "uncertain_adjudication_reason"] = u_reason

            if u_decision == "KEEP" and u_conf is not None and u_conf >= uncertain_keep_threshold:
                effective_rel = "KEEP"
            else:
                effective_rel = "DROP"
                scored_df.at[idx, "effective_relevance_decision"] = "DROP"
                print(
                    f"FLUSH_UNCERTAIN: {row['event_id']} / {row['canonical_name']} | "
                    f"initial=UNCERTAIN | adjudicated={u_decision} @ {u_conf}"
                )
                rows_since_save += 1

                if rows_since_save >= autosave_every:
                    outfile = autosave_df(scored_df, paths["scored_events_dir"])
                    print(f"AUTOSAVED: {outfile}")
                    rows_since_save = 0

                continue

        scored_df.at[idx, "effective_relevance_decision"] = effective_rel

        primary_payload = build_format_payload(row, parent_text=parent_text)
        primary_prompt = primary_prompt_template.format_map(primary_payload)

        a_s, a_c, _a_reason, a_raw = call_and_parse_sentiment(
            model_name=primary_a_model,
            prompt=primary_prompt,
            anthropic_client=anthropic_client,
            openai_client=openai_client,
            gemini_client=gemini_client,
            timeout_seconds=request_timeout,
            max_tokens=220,
            temperature=0.1,
            transport_retry_limit=retry_limit,
            semantic_retry_limit=2,
        )
        a_scum = scum_score(a_s, a_c)

        b_s, b_c, _b_reason, b_raw = call_and_parse_sentiment(
            model_name=primary_b_model,
            prompt=primary_prompt,
            anthropic_client=anthropic_client,
            openai_client=openai_client,
            gemini_client=gemini_client,
            timeout_seconds=request_timeout,
            max_tokens=220,
            temperature=0.1,
            transport_retry_limit=retry_limit,
            semantic_retry_limit=2,
        )
        b_scum = scum_score(b_s, b_c)

        if args.debug_primary_b:
            print("\n--- PRIMARY_B RAW OUTPUT ---")
            print(b_raw)
            print("--- END PRIMARY_B RAW OUTPUT ---\n")

        scored_df.at[idx, "primary_a_sentiment"] = a_s
        scored_df.at[idx, "primary_a_confidence"] = a_c
        scored_df.at[idx, "primary_a_scum"] = a_scum

        scored_df.at[idx, "primary_b_sentiment"] = b_s
        scored_df.at[idx, "primary_b_confidence"] = b_c
        scored_df.at[idx, "primary_b_scum"] = b_scum

        solomon = should_trigger_solomon(a_s, a_c, a_scum, b_s, b_c, b_scum, disagree_threshold)
        scored_df.at[idx, "solomon_triggered"] = solomon

        final_s = None
        final_c = None
        final_sc = None

        if solomon:
            tie_payload = build_format_payload(
                row,
                parent_text=parent_text,
                model_a_sentiment=a_s,
                model_a_confidence=a_c,
                model_b_sentiment=b_s,
                model_b_confidence=b_c,
            )
            tie_prompt = tiebreaker_prompt_template.format_map(tie_payload)

            t_s, t_c, t_reason, _t_raw = call_and_parse_sentiment(
                model_name=tiebreaker_model,
                prompt=tie_prompt,
                anthropic_client=anthropic_client,
                openai_client=openai_client,
                gemini_client=gemini_client,
                timeout_seconds=request_timeout,
                max_tokens=260,
                temperature=0.0,
                transport_retry_limit=retry_limit,
                semantic_retry_limit=2,
            )
            t_scum = scum_score(t_s, t_c)

            scored_df.at[idx, "tiebreaker_sentiment"] = t_s
            scored_df.at[idx, "tiebreaker_confidence"] = t_c
            scored_df.at[idx, "tiebreaker_scum"] = t_scum
            scored_df.at[idx, "tiebreaker_reason"] = t_reason

            if not is_missing_score(t_s, t_c):
                final_s = t_s
                final_c = t_c
                final_sc = t_scum
            elif not is_missing_score(a_s, a_c) and is_missing_score(b_s, b_c):
                final_s = a_s
                final_c = a_c
                final_sc = a_scum
            elif not is_missing_score(b_s, b_c) and is_missing_score(a_s, a_c):
                final_s = b_s
                final_c = b_c
                final_sc = b_scum
            else:
                final_s = None
                final_c = None
                final_sc = None
        else:
            if not is_missing_score(a_s, a_c) and not is_missing_score(b_s, b_c):
                final_s = round((a_s + b_s) / 2, 4)
                final_c = round((a_c + b_c) / 2, 4)
                final_sc = scum_score(final_s, final_c)
            elif not is_missing_score(a_s, a_c):
                final_s = a_s
                final_c = a_c
                final_sc = a_scum
            elif not is_missing_score(b_s, b_c):
                final_s = b_s
                final_c = b_c
                final_sc = b_scum

        scored_df.at[idx, "final_sentiment"] = final_s
        scored_df.at[idx, "final_confidence"] = final_c
        scored_df.at[idx, "final_scum"] = final_sc

        print(
            f"{row['event_id']} | {row['canonical_name']} | "
            f"rel={effective_rel} | A={a_scum} | B={b_scum} | "
            f"solomon={solomon} | final={final_sc}"
        )

        rows_since_save += 1
        if rows_since_save >= autosave_every:
            outfile = autosave_df(scored_df, paths["scored_events_dir"])
            print(f"AUTOSAVED: {outfile}")
            rows_since_save = 0

    outfile = autosave_df(scored_df, paths["scored_events_dir"])

    initial_keep = int((scored_df["relevance_decision"] == "KEEP").sum())
    initial_uncertain = int((scored_df["relevance_decision"] == "UNCERTAIN").sum())

    rescued_uncertain = int(
        (
            (scored_df["uncertain_adjudication_decision"] == "KEEP")
            & (
                pd.to_numeric(
                    scored_df["uncertain_adjudication_confidence"],
                    errors="coerce",
                ) >= uncertain_keep_threshold
            )
        ).sum()
    )

    final_scored = int(scored_df["final_scum"].notna().sum())
    solomon_count = int(scored_df["solomon_triggered"].astype("boolean").fillna(False).sum())

    print("\nSaved:")
    print(outfile)
    print("\nCounts:")
    print("Rows scored:", len(scored_df))
    print("Initial KEEP rows:", initial_keep)
    print("Initial UNCERTAIN rows:", initial_uncertain)
    print("Uncertain rescued:", rescued_uncertain)
    print("Final scored rows:", final_scored)
    print("Solomon triggered:", solomon_count)


if __name__ == "__main__":
    main()
