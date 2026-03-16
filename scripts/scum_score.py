import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

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
    parser.add_argument("--debug-gemini", action="store_true", help="Print raw Gemini outputs")
    return parser.parse_args()


# --------------------------------------------------
# helpers
# --------------------------------------------------

class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def latest_parquet(directory):
    files = sorted(Path(directory).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    return files[-1]


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def scum_score(sentiment, confidence):
    if sentiment is None or confidence is None:
        return None
    try:
        return round(float(sentiment) * math.sqrt(float(confidence)), 4)
    except Exception:
        return None


def parse_relevance_response(text):
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

    return decision, confidence, reason


def extract_json_block(text):
    if not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    # Remove markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # First try full text
    try:
        return json.loads(text)
    except Exception:
        pass

    # Then try to pull a JSON object block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None

    block = m.group(0)

    try:
        return json.loads(block)
    except Exception:
        pass

    # Loose fallback for single quotes
    try:
        block2 = re.sub(r"(?<!\\)'", '"', block)
        return json.loads(block2)
    except Exception:
        return None


def parse_sentiment_response(text):
    payload = extract_json_block(text)
    if not payload or not isinstance(payload, dict):
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

    return sentiment, confidence, explanation


def is_missing_score(sentiment, confidence):
    return sentiment is None or confidence is None


def should_trigger_solomon(a_s, a_c, a_scum, b_s, b_c, b_scum, thresh):
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
    model_name,
    prompt,
    timeout_seconds=20,
    max_tokens=800,
    temperature=0.2,
    retry_limit=3,
):
    last_err = None

    for _ in range(retry_limit):
        try:
            resp = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system="Return only the requested format.",
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
    model_name,
    prompt,
    timeout_seconds=20,
    max_tokens=800,
    temperature=0.2,
    retry_limit=3,
):
    last_err = None

    for _ in range(retry_limit):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "Return only the requested format."},
                    {"role": "user", "content": prompt},
                ],
                timeout=timeout_seconds,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            print(f"OpenAI error ({model_name}), retrying: {e}")
            time.sleep(1)

    raise RuntimeError(f"OpenAI failed for {model_name}: {last_err}")


def call_gemini(
    client,
    model_name,
    prompt,
    timeout_seconds=20,
    max_output_tokens=400,
    temperature=0.1,
    retry_limit=3,
):
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
    model_name,
    prompt,
    anthropic_client,
    openai_client,
    gemini_client,
    timeout_seconds=20,
    max_tokens=800,
    temperature=0.2,
    retry_limit=3,
):
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
            max_output_tokens=max(300, max_tokens),
            temperature=temperature,
            retry_limit=retry_limit,
        )

    raise ValueError(f"Unsupported model family: {model_name}")


# --------------------------------------------------
# prompt payload
# --------------------------------------------------

def build_format_payload(row, model_a_sentiment=None, model_a_confidence=None, model_b_sentiment=None, model_b_confidence=None):
    thread_title = row.get("thread_title", "") or ""
    thread_text = row.get("thread_text", "") or ""
    event_text = row.get("event_text", "") or ""
    matched_aliases = row.get("matched_aliases", "") or ""
    candidate_source = row.get("candidate_source", "") or ""
    firm_name = row.get("canonical_name", "") or ""

    return SafeDict({
        "FIRM_NAME": firm_name,
        "FUND_NAME": firm_name,
        "MATCHED_ALIASES": matched_aliases,
        "CANDIDATE_SOURCE": candidate_source,
        "THREAD_TITLE": thread_title,
        "THREAD_TEXT": thread_text,
        "THREAD_POST": thread_text,
        "PARENT_TEXT": "",
        "EVENT_TEXT": event_text,
        "MODEL_A_SENTIMENT": "" if model_a_sentiment is None else model_a_sentiment,
        "MODEL_A_CONFIDENCE": "" if model_a_confidence is None else model_a_confidence,
        "MODEL_B_SENTIMENT": "" if model_b_sentiment is None else model_b_sentiment,
        "MODEL_B_CONFIDENCE": "" if model_b_confidence is None else model_b_confidence,
        # backward-compat placeholders
        "title": thread_title,
        "post": f"Thread context:\n{thread_text}\n\nEvent text:\n{event_text}",
        "comments": "",
    })


def autosave_df(df, outdir):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outfile = Path(outdir) / f"scored_events_{ts}.parquet"
    df.to_parquet(outfile, index=False)
    return outfile


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
    primary_prompt_template = read_text(Path("prompts/scum_primary_prompt.txt"))
    tiebreaker_prompt_template = read_text(Path("prompts/scum_tiebreaker_prompt.txt"))

    relevance_model = "gpt-4o-mini"
    primary_a_model = settings["models"]["primary_a"]
    primary_b_model = settings["models"]["primary_b"]
    tiebreaker_model = settings["models"]["tiebreaker"]

    retry_limit = settings["scoring"]["retry_limit"]
    request_timeout = settings["scoring"]["request_timeout"]
    autosave_every = settings["scoring"]["autosave_every_n_rows"]
    disagree_threshold = settings["scoring"]["disagree_threshold"]

    latest_event_file = latest_parquet(paths["event_firm_dir"])
    latest_thread_file = latest_parquet(paths["thread_firm_dir"])

    event_df = pd.read_parquet(latest_event_file)
    thread_df = pd.read_parquet(latest_thread_file)

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

    if args.limit is not None:
        scored_df = scored_df.head(args.limit).copy()

    out_cols = [
        "relevance_decision",
        "relevance_confidence",
        "relevance_reason",
        "haiku_sentiment",
        "haiku_confidence",
        "haiku_scum",
        "gemini_sentiment",
        "gemini_confidence",
        "gemini_scum",
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
        # ------------------------------------------
        # 1) cheap event-level firm relevance filter
        # ------------------------------------------
        relevance_payload = build_format_payload(row)
        relevance_prompt = relevance_prompt_template.format_map(relevance_payload)

        relevance_text = call_model(
            model_name=relevance_model,
            prompt=relevance_prompt,
            anthropic_client=anthropic_client,
            openai_client=openai_client,
            gemini_client=gemini_client,
            timeout_seconds=request_timeout,
            max_tokens=250,
            temperature=0.1,
            retry_limit=retry_limit,
        )

        rel_decision, rel_conf, rel_reason = parse_relevance_response(relevance_text)

        scored_df.at[idx, "relevance_decision"] = rel_decision
        scored_df.at[idx, "relevance_confidence"] = rel_conf
        scored_df.at[idx, "relevance_reason"] = rel_reason

        if rel_decision == "DROP":
            print(f"DROP: {row['event_id']} / {row['canonical_name']}")
            rows_since_save += 1

            if rows_since_save >= autosave_every:
                outfile = autosave_df(scored_df, paths["scored_events_dir"])
                print(f"AUTOSAVED: {outfile}")
                rows_since_save = 0

            continue

        # ------------------------------------------
        # 2) primary scoring
        # ------------------------------------------
        primary_payload = build_format_payload(row)
        primary_prompt = primary_prompt_template.format_map(primary_payload)

        a_text = call_model(
            model_name=primary_a_model,
            prompt=primary_prompt,
            anthropic_client=anthropic_client,
            openai_client=openai_client,
            gemini_client=gemini_client,
            timeout_seconds=request_timeout,
            max_tokens=500,
            temperature=0.2,
            retry_limit=retry_limit,
        )
        a_s, a_c, _ = parse_sentiment_response(a_text)
        a_scum = scum_score(a_s, a_c)

        b_text = call_model(
            model_name=primary_b_model,
            prompt=primary_prompt,
            anthropic_client=anthropic_client,
            openai_client=openai_client,
            gemini_client=gemini_client,
            timeout_seconds=request_timeout,
            max_tokens=400,
            temperature=0.1,
            retry_limit=retry_limit,
        )

        if args.debug_gemini:
            print("\n--- GEMINI RAW OUTPUT ---")
            print(b_text)
            print("--- END GEMINI RAW OUTPUT ---\n")

        b_s, b_c, _ = parse_sentiment_response(b_text)
        b_scum = scum_score(b_s, b_c)

        scored_df.at[idx, "haiku_sentiment"] = a_s
        scored_df.at[idx, "haiku_confidence"] = a_c
        scored_df.at[idx, "haiku_scum"] = a_scum

        scored_df.at[idx, "gemini_sentiment"] = b_s
        scored_df.at[idx, "gemini_confidence"] = b_c
        scored_df.at[idx, "gemini_scum"] = b_scum

        # ------------------------------------------
        # 3) solomon if needed
        # ------------------------------------------
        solomon = should_trigger_solomon(a_s, a_c, a_scum, b_s, b_c, b_scum, disagree_threshold)
        scored_df.at[idx, "solomon_triggered"] = solomon

        final_s = None
        final_c = None
        final_sc = None

        if solomon:
            tie_payload = build_format_payload(
                row,
                model_a_sentiment=a_s,
                model_a_confidence=a_c,
                model_b_sentiment=b_s,
                model_b_confidence=b_c,
            )
            tie_prompt = tiebreaker_prompt_template.format_map(tie_payload)

            t_text = call_model(
                model_name=tiebreaker_model,
                prompt=tie_prompt,
                anthropic_client=anthropic_client,
                openai_client=openai_client,
                gemini_client=gemini_client,
                timeout_seconds=request_timeout,
                max_tokens=700,
                temperature=0.2,
                retry_limit=retry_limit,
            )
            t_s, t_c, t_reason = parse_sentiment_response(t_text)
            t_scum = scum_score(t_s, t_c)

            scored_df.at[idx, "tiebreaker_sentiment"] = t_s
            scored_df.at[idx, "tiebreaker_confidence"] = t_c
            scored_df.at[idx, "tiebreaker_scum"] = t_scum
            scored_df.at[idx, "tiebreaker_reason"] = t_reason

            final_s = t_s
            final_c = t_c
            final_sc = t_scum
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
            f"rel={rel_decision} | A={a_scum} | B={b_scum} | "
            f"solomon={solomon} | final={final_sc}"
        )

        rows_since_save += 1
        if rows_since_save >= autosave_every:
            outfile = autosave_df(scored_df, paths["scored_events_dir"])
            print(f"AUTOSAVED: {outfile}")
            rows_since_save = 0

    outfile = autosave_df(scored_df, paths["scored_events_dir"])

    print("\nSaved:")
    print(outfile)
    print("\nCounts:")
    print("Rows scored:", len(scored_df))
    print("Relevant rows:", (scored_df["relevance_decision"] != "DROP").sum())
    print("Solomon triggered:", int(scored_df["solomon_triggered"].astype("boolean").fillna(False).sum()))


if __name__ == "__main__":
    main()
