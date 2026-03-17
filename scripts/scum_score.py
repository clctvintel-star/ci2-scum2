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


# ==========================================================
# helper
# ==========================================================

def short_reason(txt: Any, n: int = 100) -> str:
    txt = safe_text(txt)
    if not txt:
        return ""
    txt = txt.replace("\n", " ")
    return txt[:n] + ("..." if len(txt) > n else "")


def safe_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def clamp_float(v: Any, lo: float, hi: float) -> Optional[float]:
    if v is None:
        return None
    try:
        v = float(v)
    except Exception:
        return None
    return max(lo, min(hi, v))


def scum_score(sentiment: Optional[float], confidence: Optional[float]) -> Optional[float]:
    if sentiment is None or confidence is None:
        return None
    try:
        return round(float(sentiment) * math.sqrt(float(confidence)), 4)
    except Exception:
        return None


def normalize_source_bucket(candidate_source: Any) -> str:
    txt = safe_text(candidate_source).lower()
    if txt == "direct_event_match":
        return "direct"
    if txt == "thread_context":
        return "context"
    return "unknown"


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


# ==========================================================
# args
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="SCUM2 event-level scorer")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--firm", type=str, default=None)

    parser.add_argument(
        "--source-bucket",
        type=str,
        choices=["direct", "context"],
        default=None,
    )

    parser.add_argument("--sample-per-firm", type=int, default=None)

    parser.add_argument("--event-file", type=str, default=None)
    parser.add_argument("--thread-file", type=str, default=None)

    parser.add_argument("--show-latest-files", action="store_true")
    parser.add_argument("--debug-primary-b", action="store_true")
    parser.add_argument("--debug-save-raw", action="store_true")

    return parser.parse_args()


# ==========================================================
# file helpers
# ==========================================================

def latest_parquet(directory: str, prefix: Optional[str] = None) -> Path:
    pattern = "*.parquet" if prefix is None else f"{prefix}_*.parquet"
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory} matching {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


def resolve_input_file(explicit_path: Optional[str], directory: str, prefix: str) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    return latest_parquet(directory, prefix)


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def autosave_df(df: pd.DataFrame, outdir: str, prefix: str = "scored_events") -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outfile = Path(outdir) / f"{prefix}_{ts}.parquet"
    df.to_parquet(outfile, index=False)
    return outfile


def print_latest_files(paths: Dict[str, str]) -> None:
    latest_event = latest_parquet(paths["event_firm_dir"], prefix="event_firm_pairs")
    latest_thread = latest_parquet(paths["thread_firm_dir"], prefix="thread_firm_pairs")
    print("Latest thread file:", latest_thread.name)
    print("Latest event file :", latest_event.name)


# ==========================================================
# parsing
# ==========================================================

def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def coerce_incomplete_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None

    sent_match = re.search(r'"sentiment"\s*:\s*(null|-?\d+(?:\.\d+)?)', text)
    conf_match = re.search(r'"confidence"\s*:\s*(null|\d+(?:\.\d+)?)', text)
    expl_match = re.search(r'"(explanation|reason|rationale)"\s*:\s*"([^"]*)', text)

    if not sent_match or not conf_match:
        return None

    payload: Dict[str, Any] = {}

    s = sent_match.group(1)
    c = conf_match.group(1)

    payload["sentiment"] = None if s == "null" else float(s)
    payload["confidence"] = None if c == "null" else float(c)
    payload["explanation"] = expl_match.group(2).strip() if expl_match else None

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

    m = re.search(r"\{.*\}", text, re.S)
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

    sentiment = clamp_float(sentiment, -1.0, 1.0)
    confidence = clamp_float(confidence, 0.0, 1.0)
    explanation = safe_text(explanation) or None

    return sentiment, confidence, explanation


# ==========================================================
# model selection logic
# ==========================================================

def is_valid_score(s: Optional[float], c: Optional[float]) -> bool:
    return s is not None and c is not None


def should_trigger_solomon(
    a_s: Optional[float],
    a_c: Optional[float],
    a_sc: Optional[float],
    b_s: Optional[float],
    b_c: Optional[float],
    b_sc: Optional[float],
    thresh: float,
) -> bool:
    # Only adjudicate when BOTH models produced valid scores.
    if not is_valid_score(a_s, a_c):
        return False
    if not is_valid_score(b_s, b_c):
        return False
    if a_sc is None or b_sc is None:
        return False
    if a_s * b_s < 0:
        return True
    if abs(a_sc - b_sc) > thresh:
        return True
    return False


# ==========================================================
# clients
# ==========================================================

def build_clients(env):
    anthropic_client = Anthropic(api_key=env["ANTHROPIC_API_KEY"])
    openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])
    gemini_client = genai.Client(api_key=env["GOOGLE_API_KEY"])
    return anthropic_client, openai_client, gemini_client


# ==========================================================
# model calls
# ==========================================================

def call_anthropic(
    client,
    model_name: str,
    prompt: str,
    max_tokens: int = 250,
    temperature: float = 0.1,
    retry_limit: int = 3,
) -> str:
    last_err = None
    for _ in range(retry_limit):
        try:
            r = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise RuntimeError(f"Anthropic failed for {model_name}: {last_err}")


def call_openai(
    client,
    model_name: str,
    prompt: str,
    max_tokens: int = 250,
    temperature: float = 0.1,
    retry_limit: int = 3,
) -> str:
    last_err = None
    for _ in range(retry_limit):
        try:
            r = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise RuntimeError(f"OpenAI failed for {model_name}: {last_err}")


def call_gemini(
    client,
    model_name: str,
    prompt: str,
    max_tokens: int = 250,
    temperature: float = 0.1,
    retry_limit: int = 3,
) -> str:
    last_err = None
    for _ in range(retry_limit):
        try:
            r = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "response_mime_type": "application/json",
                },
            )
            text = getattr(r, "text", None)
            return text or str(r)
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise RuntimeError(f"Gemini failed for {model_name}: {last_err}")


def call_model(
    model_name: str,
    prompt: str,
    anthropic_client,
    openai_client,
    gemini_client,
    max_tokens: int = 250,
    temperature: float = 0.1,
    retry_limit: int = 3,
) -> str:
    if model_name.startswith("claude"):
        return call_anthropic(
            anthropic_client, model_name, prompt,
            max_tokens=max_tokens, temperature=temperature, retry_limit=retry_limit
        )

    if model_name.startswith("gpt"):
        return call_openai(
            openai_client, model_name, prompt,
            max_tokens=max_tokens, temperature=temperature, retry_limit=retry_limit
        )

    if model_name.startswith("gemini"):
        return call_gemini(
            gemini_client, model_name, prompt,
            max_tokens=max_tokens, temperature=temperature, retry_limit=retry_limit
        )

    raise ValueError(f"Unsupported model family: {model_name}")


def call_and_parse_sentiment(
    model_name: str,
    prompt: str,
    anthropic_client,
    openai_client,
    gemini_client,
    max_tokens: int = 250,
    temperature: float = 0.1,
    retry_limit: int = 3,
    semantic_retry_limit: int = 2,
) -> Tuple[Optional[float], Optional[float], Optional[str], str]:
    last_raw = ""

    for attempt in range(semantic_retry_limit):
        effective_prompt = prompt
        if attempt > 0:
            effective_prompt = (
                prompt
                + "\n\nIMPORTANT: Return one complete valid JSON object only with keys "
                  '"sentiment", "confidence", and "explanation". '
                  "No markdown. No extra text."
            )

        raw = call_model(
            model_name=model_name,
            prompt=effective_prompt,
            anthropic_client=anthropic_client,
            openai_client=openai_client,
            gemini_client=gemini_client,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_limit=retry_limit,
        )

        last_raw = raw
        sentiment, confidence, reason = parse_sentiment_response(raw)
        if confidence is not None:
            return sentiment, confidence, reason, raw

    return None, None, None, last_raw


# ==========================================================
# prompt payload
# ==========================================================

def build_format_payload(
    row: pd.Series,
    model_a_sentiment=None,
    model_a_confidence=None,
    model_b_sentiment=None,
    model_b_confidence=None,
) -> SafeDict:
    return SafeDict({
        "FIRM_NAME": safe_text(row.get("canonical_name")),
        "FUND_NAME": safe_text(row.get("canonical_name")),
        "SOURCE_BUCKET": safe_text(row.get("source_bucket")),
        "CANDIDATE_SOURCE": safe_text(row.get("candidate_source")),
        "MATCHED_ALIASES": safe_text(row.get("matched_aliases")),
        "EVENT_TEXT": safe_text(row.get("event_text")),
        "THREAD_TEXT": safe_text(row.get("thread_text")),
        "THREAD_TITLE": safe_text(row.get("thread_title")),
        "MODEL_A_SENTIMENT": "" if model_a_sentiment is None else model_a_sentiment,
        "MODEL_A_CONFIDENCE": "" if model_a_confidence is None else model_a_confidence,
        "MODEL_B_SENTIMENT": "" if model_b_sentiment is None else model_b_sentiment,
        "MODEL_B_CONFIDENCE": "" if model_b_confidence is None else model_b_confidence,
    })


# ==========================================================
# dataframe setup
# ==========================================================

def prepare_scoring_df(event_df: pd.DataFrame, thread_df: pd.DataFrame) -> pd.DataFrame:
    thread_merge_cols = [
        "post_id",
        "canonical_name",
        "thread_title",
        "thread_text",
        "llm_rationale",
        "mention_type",
        "candidate_strength",
        "thread_direct_match",
    ]

    available_thread_cols = [c for c in thread_merge_cols if c in thread_df.columns]

    if {"post_id", "canonical_name"}.issubset(set(available_thread_cols)):
        df = event_df.merge(
            thread_df[available_thread_cols].drop_duplicates(subset=["post_id", "canonical_name"]),
            on=["post_id", "canonical_name"],
            how="left",
        )
    else:
        df = event_df.copy()

    df["source_bucket"] = df["candidate_source"].apply(normalize_source_bucket)

    out_cols = [
        "primary_a_sentiment",
        "primary_a_confidence",
        "primary_a_scum",
        "primary_a_reason",

        "primary_b_sentiment",
        "primary_b_confidence",
        "primary_b_scum",
        "primary_b_reason",

        "tiebreaker_sentiment",
        "tiebreaker_confidence",
        "tiebreaker_scum",
        "tiebreaker_reason",

        "final_sentiment",
        "final_confidence",
        "final_scum",
        "final_reason",
        "final_reason_source",

        "solomon_triggered",

        "primary_a_raw",
        "primary_b_raw",
        "tiebreaker_raw",
    ]

    for c in out_cols:
        if c not in df.columns:
            df[c] = None

    return df


def apply_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()

    if args.firm:
        out = out[out["canonical_name"] == args.firm].copy()

    if args.source_bucket:
        out = out[out["source_bucket"] == args.source_bucket].copy()

    if args.sample_per_firm is not None:
        out = (
            out.groupby("canonical_name", group_keys=False)
               .head(args.sample_per_firm)
               .copy()
        )

    if args.limit is not None:
        out = out.head(args.limit).copy()

    return out


# ==========================================================
# main
# ==========================================================

def main():
    args = parse_args()
    _, paths, settings = load_configs()
    env = load_env(paths["env_path"])

    if args.show_latest_files:
        print_latest_files(paths)
        return

    ensure_dir(paths["scored_events_dir"])

    anthropic_client, openai_client, gemini_client = build_clients(env)

    primary_prompt = read_text(Path("prompts/scum_primary_prompt.txt"))
    tie_prompt = read_text(Path("prompts/scum_tiebreaker_prompt.txt"))

    event_file = resolve_input_file(args.event_file, paths["event_firm_dir"], "event_firm_pairs")
    thread_file = resolve_input_file(args.thread_file, paths["thread_firm_dir"], "thread_firm_pairs")

    event_df = pd.read_parquet(event_file)
    thread_df = pd.read_parquet(thread_file)

    df = prepare_scoring_df(event_df, thread_df)
    df = apply_filters(df, args).reset_index(drop=True)

    print("Using input files:")
    print("EVENT :", event_file)
    print("THREAD:", thread_file)
    print()
    print("Rows queued:", len(df))

    if "source_bucket" in df.columns:
        print("Source bucket counts:")
        print(df["source_bucket"].value_counts(dropna=False))
        print()

    retry_limit = settings.get("scoring", {}).get("retry_limit", 3)
    disagree_threshold = settings.get("scoring", {}).get("disagree_threshold", 0.2)
    autosave_every = settings.get("scoring", {}).get("autosave_every_n_rows", 50)

    primary_a_model = settings["models"]["score"]["primary_a"]
    primary_b_model = settings["models"]["score"]["primary_b"]
    tiebreaker_model = settings["models"]["score"]["tiebreaker"]

    rows_since_save = 0

    for idx, row in df.iterrows():
        payload = build_format_payload(row)
        prompt = primary_prompt.format_map(payload)

        a_s, a_c, a_reason, a_raw = call_and_parse_sentiment(
            primary_a_model,
            prompt,
            anthropic_client,
            openai_client,
            gemini_client,
            retry_limit=retry_limit,
        )

        b_s, b_c, b_reason, b_raw = call_and_parse_sentiment(
            primary_b_model,
            prompt,
            anthropic_client,
            openai_client,
            gemini_client,
            retry_limit=retry_limit,
        )

        if args.debug_primary_b:
            print("\n--- PRIMARY_B RAW OUTPUT ---")
            print(b_raw)
            print("--- END PRIMARY_B RAW OUTPUT ---\n")

        a_sc = scum_score(a_s, a_c)
        b_sc = scum_score(b_s, b_c)

        df.at[idx, "primary_a_sentiment"] = a_s
        df.at[idx, "primary_a_confidence"] = a_c
        df.at[idx, "primary_a_scum"] = a_sc
        df.at[idx, "primary_a_reason"] = a_reason

        df.at[idx, "primary_b_sentiment"] = b_s
        df.at[idx, "primary_b_confidence"] = b_c
        df.at[idx, "primary_b_scum"] = b_sc
        df.at[idx, "primary_b_reason"] = b_reason

        if args.debug_save_raw:
            df.at[idx, "primary_a_raw"] = a_raw
            df.at[idx, "primary_b_raw"] = b_raw

        solomon = should_trigger_solomon(a_s, a_c, a_sc, b_s, b_c, b_sc, disagree_threshold)
        df.at[idx, "solomon_triggered"] = solomon

        final_s = None
        final_c = None
        final_sc = None
        final_reason = None
        final_reason_source = "none"

        a_valid = is_valid_score(a_s, a_c)
        b_valid = is_valid_score(b_s, b_c)

        if a_valid and b_valid and solomon:
            tie_payload = build_format_payload(
                row,
                model_a_sentiment=a_s,
                model_a_confidence=a_c,
                model_b_sentiment=b_s,
                model_b_confidence=b_c,
            )
            t_prompt = tie_prompt.format_map(tie_payload)

            t_s, t_c, t_reason, t_raw = call_and_parse_sentiment(
                tiebreaker_model,
                t_prompt,
                anthropic_client,
                openai_client,
                gemini_client,
                retry_limit=retry_limit,
            )

            t_sc = scum_score(t_s, t_c)

            df.at[idx, "tiebreaker_sentiment"] = t_s
            df.at[idx, "tiebreaker_confidence"] = t_c
            df.at[idx, "tiebreaker_scum"] = t_sc
            df.at[idx, "tiebreaker_reason"] = t_reason

            if args.debug_save_raw:
                df.at[idx, "tiebreaker_raw"] = t_raw

            if is_valid_score(t_s, t_c):
                final_s = t_s
                final_c = t_c
                final_sc = t_sc
                final_reason = t_reason
                final_reason_source = "tiebreaker"
            else:
                final_s = round((a_s + b_s) / 2, 4)
                final_c = round((a_c + b_c) / 2, 4)
                final_sc = scum_score(final_s, final_c)
                final_reason = f"A:{safe_text(a_reason)} | B:{safe_text(b_reason)}"
                final_reason_source = "primary_avg_tiebreaker_failed"

        elif a_valid and b_valid:
            final_s = round((a_s + b_s) / 2, 4)
            final_c = round((a_c + b_c) / 2, 4)
            final_sc = scum_score(final_s, final_c)
            final_reason = f"A:{safe_text(a_reason)} | B:{safe_text(b_reason)}"
            final_reason_source = "primary_avg"

        elif a_valid and not b_valid:
            final_s = a_s
            final_c = a_c
            final_sc = a_sc
            final_reason = a_reason
            final_reason_source = "primary_a_only"

        elif b_valid and not a_valid:
            final_s = b_s
            final_c = b_c
            final_sc = b_sc
            final_reason = b_reason
            final_reason_source = "primary_b_only"

        else:
            final_s = None
            final_c = None
            final_sc = None
            final_reason = None
            final_reason_source = "none"

        df.at[idx, "final_sentiment"] = final_s
        df.at[idx, "final_confidence"] = final_c
        df.at[idx, "final_scum"] = final_sc
        df.at[idx, "final_reason"] = final_reason
        df.at[idx, "final_reason_source"] = final_reason_source

        event_key = row.get("event_id", idx)

        print(
            f"[{idx + 1}/{len(df)}] "
            f"{event_key} | {row['canonical_name']} | source={row['source_bucket']} | "
            f"A={a_sc} | B={b_sc} | solomon={solomon} | final={final_sc} | "
            f"final_source={final_reason_source} | "
            f"A_reason={short_reason(a_reason)} | "
            f"B_reason={short_reason(b_reason)}"
        )

        rows_since_save += 1
        if rows_since_save >= autosave_every:
            outfile = autosave_df(df, paths["scored_events_dir"])
            print(f"AUTOSAVED: {outfile}")
            rows_since_save = 0

    outfile = autosave_df(df, paths["scored_events_dir"])

    print("\nSaved:", outfile)
    print("\nFinal scored rows:", int(df["final_scum"].notna().sum()))
    print("\nFinal reason sources:")
    print(df["final_reason_source"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
