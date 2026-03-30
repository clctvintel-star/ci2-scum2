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
# constants
# ==========================================================

COMPLETED_REASON_SOURCES = {
    "prefilter_drop",
    "primary_avg",
    "tiebreaker",
    "primary_failure",
    "primary_avg_tiebreaker_failed",
    "firm_link_category",
    "firm_link_drop",
    "primary_a_only",
    "primary_b_only",
}

STATEFUL_COLUMNS = [
    "title_inheritance_rescue",
    "prefilter_decision",
    "prefilter_confidence",
    "prefilter_reason",
    "prefilter_source",
    "prefilter_raw",
    "primary_a_sentiment",
    "primary_a_confidence",
    "primary_a_scum",
    "primary_a_reason",
    "primary_a_status",
    "primary_b_sentiment",
    "primary_b_confidence",
    "primary_b_scum",
    "primary_b_reason",
    "primary_b_status",
    "tiebreaker_sentiment",
    "tiebreaker_confidence",
    "tiebreaker_scum",
    "tiebreaker_reason",
    "tiebreaker_status",
    "final_sentiment",
    "final_confidence",
    "final_scum",
    "final_reason",
    "final_reason_source",
    "solomon_triggered",
    "primary_a_raw",
    "primary_b_raw",
    "tiebreaker_raw",
    "firm_link_label",
    "firm_link_confidence",
    "firm_link_reason",
    "firm_link_raw",
]

CHECKPOINT_COLUMNS = [
    "event_id",
    "canonical_name",
    "_row_key",
] + STATEFUL_COLUMNS


# ==========================================================
# helper utils
# ==========================================================

def safe_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def short_reason(txt: Any, n: int = 120) -> str:
    txt = safe_text(txt)
    if not txt:
        return ""
    txt = txt.replace("\n", " ")
    return txt[:n] + ("..." if len(txt) > n else "")


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
        return round(float(sentiment) * (0.5 + 0.5 * float(confidence)), 4)
    except Exception:
        return None


def normalize_source_bucket(candidate_source: Any) -> str:
    txt = safe_text(candidate_source).lower()
    if txt == "direct_event_match":
        return "direct"
    if txt == "thread_context":
        return "context"
    return "unknown"


def normalize_text_for_prefilter(text: Any) -> str:
    text = safe_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", safe_text(text)))

def estimate_signal_strength(row: pd.Series) -> int:
    text = safe_text(row.get("event_text"))
    title = safe_text(row.get("thread_title"))
    firm = row.get("canonical_name")

    score = 0

    # direct mention in event text = strongest
    if text_contains_firm_name(text, firm):
        score += 2

    # mention only in thread title = weaker context
    if text_contains_firm_name(title, firm):
        score += 1

    # longer content = more likely meaningful
    if token_count(text) > 20:
        score += 1

    return score  # 0–4

def current_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def normalize_for_contains(text: Any) -> str:
    text = safe_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} " if text else " "


def text_contains_firm_name(text: Any, firm_name: Any) -> bool:
    norm_text = normalize_for_contains(text)
    norm_firm = normalize_for_contains(firm_name).strip()
    if not norm_firm:
        return False
    return f" {norm_firm} " in norm_text


def should_rescue_by_thread_title(row: pd.Series) -> bool:
    source_bucket = safe_text(row.get("source_bucket")).lower()
    if source_bucket != "context":
        return False
    return text_contains_firm_name(row.get("thread_title"), row.get("canonical_name"))


def build_row_key(df: pd.DataFrame) -> pd.Series:
    event_ids = df["event_id"].fillna("").astype(str)
    firm_names = df["canonical_name"].fillna("").astype(str)
    return event_ids + "||" + firm_names


def is_completed_df(df: pd.DataFrame) -> pd.Series:
    final_reason_source = df.get("final_reason_source")
    final_scum = df.get("final_scum")

    if final_reason_source is None:
        final_reason_source = pd.Series([None] * len(df), index=df.index)
    if final_scum is None:
        final_scum = pd.Series([None] * len(df), index=df.index)

    return final_scum.notna() | final_reason_source.isin(COMPLETED_REASON_SOURCES)


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
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default="scored_events")

    parser.add_argument("--show-latest-files", action="store_true")
    parser.add_argument("--debug-primary-a", action="store_true")
    parser.add_argument("--debug-primary-b", action="store_true")
    parser.add_argument("--debug-tiebreaker", action="store_true")
    parser.add_argument("--debug-save-raw", action="store_true")
    parser.add_argument("--disable-llm-prefilter", action="store_true")
    parser.add_argument("--disable-heuristic-prefilter", action="store_true")

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

def atomic_write_parquet(df: pd.DataFrame, outfile: Path) -> Path:
    ensure_dir(outfile.parent)
    df.to_parquet(outfile, index=False)
    return outfile


def autosave_df(df: pd.DataFrame, outdir: str, prefix: str = "scored_events") -> Path:
    outfile = Path(outdir) / f"{prefix}_latest.parquet"
    clean_df = df.drop(columns=["_row_key"], errors="ignore")
    return atomic_write_parquet(clean_df, outfile)


def save_df_clean(df: pd.DataFrame, outdir: str, prefix: str = "scored_events") -> Path:
    return autosave_df(df, outdir, prefix=prefix)


def save_final_df(df: pd.DataFrame, outdir: str, prefix: str = "scored_events") -> Path:
    outfile = Path(outdir) / f"{prefix}_{current_utc_stamp()}.parquet"
    clean_df = df.drop(columns=["_row_key"], errors="ignore")
    return atomic_write_parquet(clean_df, outfile)


def save_checkpoint_df(df: pd.DataFrame, outdir: str, prefix: str = "scored_events") -> Path:
    outfile = Path(outdir) / f"{prefix}_checkpoint_latest.parquet"
    cols_present = [c for c in CHECKPOINT_COLUMNS if c in df.columns]
    slim_df = df[cols_present].copy()
    return atomic_write_parquet(slim_df, outfile)


def print_latest_files(paths: Dict[str, str]) -> None:
    latest_event = latest_parquet(paths["event_firm_dir"], prefix="event_firm_pairs")
    latest_thread = latest_parquet(paths["thread_firm_dir"], prefix="thread_firm_pairs")
    print("Latest thread file:", latest_thread.name)
    print("Latest event file :", latest_event.name)


def load_resume_file(resume_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not resume_path:
        return None
    p = Path(resume_path)
    if not p.exists():
        raise FileNotFoundError(f"Resume file not found: {p}")
    return pd.read_parquet(p).copy()


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
            return {k.lower(): v for k, v in payload.items()}
  
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.S)
    if m:
        block = m.group(0)

        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                return {k.lower(): v for k, v in payload.items()}
        except Exception:
            pass

        try:
            block2 = re.sub(r"(?<!\\)'", '"', block)
            payload = json.loads(block2)
            if isinstance(payload, dict):
                return {k.lower(): v for k, v in payload.items()}
        except Exception:
            pass

    repaired = coerce_incomplete_json(text)
    if isinstance(repaired, dict):
        return repaired

    return None


def parse_sentiment_response(text: Any) -> Tuple[Optional[float], Optional[float], Optional[str], str]:
    payload = extract_json_payload(text) or {}
    if not payload:
        return None, None, None, "parse_failure"

    sentiment = payload.get("sentiment")
    confidence = payload.get("confidence")
    explanation = payload.get("explanation") or payload.get("reason") or payload.get("rationale")

    sentiment = clamp_float(sentiment, -1.0, 1.0)
    confidence = clamp_float(confidence, 0.0, 1.0)
    explanation = safe_text(explanation) or None

    if sentiment is None and confidence == 0.0:
        sentiment = 0.0
        confidence = 0.10
        return sentiment, confidence, explanation, "neutralized_abstention"

    if sentiment is None or confidence is None:
        return None, None, explanation, "parse_failure"

    return sentiment, confidence, explanation, "valid"


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
        confidence = clamp_float(m.group(1), 0.0, 1.0)

    m = re.search(r"Reason:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()

    if decision in {"KEEP", "DROP", "UNCERTAIN"}:
        return decision, confidence, reason

    payload = extract_json_payload(text)
    if isinstance(payload, dict):
        d = safe_text(payload.get("decision")).upper()
        if d in {"KEEP", "DROP", "UNCERTAIN"}:
            decision = d
            confidence = clamp_float(payload.get("confidence"), 0.0, 1.0)
            reason = safe_text(payload.get("reason") or payload.get("explanation") or payload.get("rationale")) or None

    return decision, confidence, reason
    
def parse_firm_link_response(text: Any) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    if not isinstance(text, str):
        return None, None, None

    label = None
    confidence = None
    reason = None

    m = re.search(r"Label:\s*(direct|strong_parent_link|category|ambient|unrelated)", text, flags=re.IGNORECASE)
    if m:
        label = m.group(1).lower()

    m = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if m:
        confidence = clamp_float(m.group(1), 0.0, 1.0)

    m = re.search(r"Reason:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()

    if label in {"direct", "strong_parent_link", "category", "ambient", "unrelated"}:
        return label, confidence, reason

    payload = extract_json_payload(text)
    if isinstance(payload, dict):
        raw_label = safe_text(payload.get("label")).lower()
        if raw_label in {"direct", "strong_parent_link", "category", "ambient", "unrelated"}:
            label = raw_label
            confidence = clamp_float(payload.get("confidence"), 0.0, 1.0)
            reason = safe_text(payload.get("reason") or payload.get("explanation") or payload.get("rationale")) or None

    return label, confidence, reason


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
                response_format={"type": "json_object"},  # 🔥 THIS IS THE FIX
                messages=[{"role": "user", "content": prompt}],
            )

            content = r.choices[0].message.content

            if isinstance(content, dict):
                return json.dumps(content)

            return content or ""

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

            # 1) clean text shortcut
            if hasattr(r, "text") and r.text:
                return r.text

            # 2) walk candidates/parts safely
            if hasattr(r, "candidates") and r.candidates:
                texts = []
                for cand in r.candidates:
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", None) if content is not None else None
                    if parts:
                        for part in parts:
                            txt = getattr(part, "text", None)
                            if txt:
                                texts.append(txt)
                if texts:
                    return "\n".join(texts).strip()

            # 3) last resort: stringify whole response for debug
            raise ValueError(f"No valid text in Gemini response: {r}")

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
            anthropic_client,
            model_name,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_limit=retry_limit,
        )

    if model_name.startswith("gpt"):
        return call_openai(
            openai_client,
            model_name,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_limit=retry_limit,
        )

    if model_name.startswith("gemini"):
        return call_gemini(
            gemini_client,
            model_name,
            prompt,
            max_tokens=max_tokens,
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
    max_tokens: int = 250,
    temperature: float = 0.1,
    retry_limit: int = 3,
    semantic_retry_limit: int = 3,
) -> Tuple[Optional[float], Optional[float], Optional[str], str, str]:
    last_raw = ""

    for attempt in range(semantic_retry_limit):
        effective_prompt = prompt
        if attempt > 0:
            effective_prompt = (
                prompt
                + "\n\nIMPORTANT: Return ONLY a valid JSON object with keys "
                + '"sentiment", "confidence", and "explanation". '
                + "Do NOT say 'Here is the JSON requested'. "
                + "Do NOT use markdown fences. "
                + "Do NOT output any text before or after the JSON."
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

        last_raw = raw if isinstance(raw, str) else ""

        cleaned = last_raw.strip()

        cleaned = re.sub(
            r"^here is the json requested:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = strip_code_fences(cleaned).strip()

        if not cleaned or "{" not in cleaned:
            continue

        sentiment, confidence, reason, status = parse_sentiment_response(cleaned)

        if status in {"valid", "neutralized_abstention"}:
            return sentiment, confidence, reason, last_raw, status

    return None, None, None, last_raw, "parse_failure"
    last_raw = ""

    for attempt in range(semantic_retry_limit):
        effective_prompt = prompt
        if attempt > 0:
            effective_prompt = (
                prompt
                + "\n\nIMPORTANT: Return ONLY a valid JSON object with keys "
                + '"sentiment", "confidence", and "explanation". '
                + "No markdown. No text before or after."
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
        sentiment, confidence, reason, status = parse_sentiment_response(raw)

        if status in {"valid", "neutralized_abstention"}:
            return sentiment, confidence, reason, raw, status

    return None, None, None, last_raw, "parse_failure"


def call_and_parse_relevance(
    model_name: str,
    prompt: str,
    anthropic_client,
    openai_client,
    gemini_client,
    max_tokens: int = 180,
    temperature: float = 0.0,
    retry_limit: int = 3,
    semantic_retry_limit: int = 2,
) -> Tuple[Optional[str], Optional[float], Optional[str], str]:
    last_raw = ""

    for attempt in range(semantic_retry_limit):
        effective_prompt = prompt
        if attempt > 0:
            effective_prompt = (
                prompt
                + "\n\nIMPORTANT: Return only either:\n"
                + "Decision: KEEP|DROP|UNCERTAIN\n"
                + "Confidence: 0.0-1.0\n"
                + "Reason: ...\n"
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
        decision, confidence, reason = parse_relevance_response(raw)
        if decision in {"KEEP", "DROP", "UNCERTAIN"}:
            return decision, confidence, reason, raw

    return None, None, None, last_raw

def call_and_parse_firm_link(
    prompt: str,
    anthropic_client,
    openai_client,
    gemini_client,
    model_name: str,
    max_tokens: int = 120,
    temperature: float = 0.0,
    retry_limit: int = 3,
    semantic_retry_limit: int = 2,
) -> Tuple[Optional[str], Optional[float], Optional[str], str]:
    last_raw = ""

    for attempt in range(semantic_retry_limit):
        effective_prompt = prompt
        if attempt > 0:
            effective_prompt = (
                prompt
                + "\n\nIMPORTANT: Return only exactly this format:\n"
                + "Label: direct or strong_parent_link or category or ambient or unrelated\n"
                + "Confidence: 0.0-1.0\n"
                + "Reason: ...\n"
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
        label, confidence, reason = parse_firm_link_response(raw)

        if label in {"direct", "strong_parent_link", "category", "ambient", "unrelated"}:
            return label, confidence, reason, raw

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
        "PARENT_TEXT": safe_text(row.get("parent_text")),
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

    for c in STATEFUL_COLUMNS:
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
            .apply(lambda x: x.sample(n=min(len(x), args.sample_per_firm), random_state=42))
            .reset_index(drop=True)
        )

    return out


def overlay_resume_state(df: pd.DataFrame, resume_df: pd.DataFrame) -> pd.DataFrame:
    resume_df = resume_df.copy()
    resume_df["_row_key"] = build_row_key(resume_df)
    resume_df = resume_df.drop_duplicates(subset=["_row_key"], keep="last").copy()

    completed_mask = is_completed_df(resume_df)
    resume_done = resume_df.loc[
        completed_mask,
        ["_row_key"] + [c for c in STATEFUL_COLUMNS if c in resume_df.columns],
    ].copy()

    if resume_done.empty:
        return df

    overlay_cols = [c for c in STATEFUL_COLUMNS if c in resume_done.columns]
    overlay = resume_done.set_index("_row_key")[overlay_cols]

    df = df.copy()
    df_idx = df.set_index("_row_key")

    common_keys = df_idx.index.intersection(overlay.index)
    if len(common_keys) == 0:
        return df

    for col in overlay_cols:
        df_idx.loc[common_keys, col] = overlay.loc[common_keys, col]

    return df_idx.reset_index()


# ==========================================================
# prefilter
# ==========================================================

def heuristic_prefilter(row: pd.Series) -> Tuple[Optional[str], Optional[float], Optional[str], str]:
    text = normalize_text_for_prefilter(row.get("event_text"))
    tok_n = token_count(text)

    if not text:
        return "DROP", 1.0, "Empty event text.", "heuristic"

    if text in {"[deleted]", "[removed]", "deleted", "removed"}:
        return "DROP", 1.0, f"Event text is {text}.", "heuristic"

    if text in {"ok", "yes", "no", "thanks", "same", "bump"}:
        return "DROP", 0.99, "Low-information acknowledgment.", "heuristic"

    if len(text) < 12 or tok_n <= 2:
        return "DROP", 0.98, f"Event text too short for reputational scoring (len={len(text)}, tokens={tok_n}).", "heuristic"

    return None, None, None, "heuristic"


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
    
    primary_prompt = read_text(Path("prompts/context_prompt.txt"))
    tie_prompt = read_text(Path("prompts/context_tiebreaker_prompt.txt"))
    
    relevance_prompt = None
    firm_link_prompt = None

    event_file = resolve_input_file(args.event_file, paths["event_firm_dir"], "event_firm_pairs")
    thread_file = resolve_input_file(args.thread_file, paths["thread_firm_dir"], "thread_firm_pairs")

    event_df = pd.read_parquet(event_file)
    thread_df = pd.read_parquet(thread_file)

    df = prepare_scoring_df(event_df, thread_df)
    df = apply_filters(df, args).reset_index(drop=True)
    df["_row_key"] = build_row_key(df)
    df["signal_strength"] = df.apply(estimate_signal_strength, axis=1)

    resume_df = load_resume_file(args.resume_from)
    if resume_df is not None:
        if "event_id" not in resume_df.columns or "canonical_name" not in resume_df.columns:
            raise ValueError("Resume file must contain event_id and canonical_name columns")

        print(f"Resume file: {args.resume_from}")
        print(f"Resume rows loaded: {len(resume_df)}")

        df = overlay_resume_state(df, resume_df)
        completed_mask = is_completed_df(df)
        already_done = int(completed_mask.sum())
        pending_indices = df.index[~completed_mask].tolist()

        print(f"Already completed rows found: {already_done}")
        print(f"Rows remaining to score before limit: {len(pending_indices)}")
    else:
        pending_indices = df.index.tolist()
        already_done = 0

    if args.limit is not None:
        pending_indices = pending_indices[:args.limit]

    print("Using input files:")
    print("EVENT :", event_file)
    print("THREAD:", thread_file)
    print()
    print("Rows queued:", len(df))
    print("Already done:", already_done)
    print("Rows pending this run:", len(pending_indices))

    if "source_bucket" in df.columns:
        print("Source bucket counts:")
        print(df["source_bucket"].value_counts(dropna=False))
        print()

    if len(pending_indices) == 0:
        print("Nothing left to score.")
        checkpoint_outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
        latest_outfile = save_df_clean(df, paths["scored_events_dir"], prefix=args.output_prefix)
        print("Latest checkpoint save:", checkpoint_outfile)
        print("Latest rolling save:", latest_outfile)
        return

    retry_limit = settings.get("scoring", {}).get("retry_limit", 3)
    disagree_threshold = settings.get("scoring", {}).get("disagree_threshold", 0.2)

    checkpoint_every = 10
    full_autosave_every = 200
    
    primary_a_model = settings["models"]["score"]["primary_a"]
    primary_b_model = settings["models"]["score"]["primary_b"]
    tiebreaker_model = settings["models"]["score"]["tiebreaker"]
    relevance_model = settings["models"]["score"].get("relevance_filter", primary_b_model)
    firm_link_model = settings["models"]["score"].get("firm_link_filter", "gpt-4o-mini")

    rows_since_checkpoint = 0
    rows_since_full_autosave = 0
    total_pending = len(pending_indices)

    for counter, idx in enumerate(pending_indices, start=1):
        row = df.loc[idx]
        event_key = row.get("event_id", idx)
        payload = build_format_payload(row)

        title_rescue = should_rescue_by_thread_title(row)
        df.at[idx, "title_inheritance_rescue"] = bool(title_rescue)

        if not args.disable_heuristic_prefilter:
            pre_decision, pre_conf, pre_reason, pre_source = heuristic_prefilter(row)

            if pre_decision == "DROP":
                df.at[idx, "prefilter_decision"] = pre_decision
                df.at[idx, "prefilter_confidence"] = pre_conf
                df.at[idx, "prefilter_reason"] = pre_reason
                df.at[idx, "prefilter_source"] = pre_source
                df.at[idx, "final_reason"] = pre_reason
                df.at[idx, "final_reason_source"] = "prefilter_drop"

                print(
                    f"[{counter}/{total_pending}] {event_key} | {row['canonical_name']} | "
                    f"source={row['source_bucket']} | PREFILTER_DROP | "
                    f"reason={short_reason(pre_reason)}"
                )

                rows_since_checkpoint += 1
                rows_since_full_autosave += 1

                if rows_since_checkpoint >= checkpoint_every:
                    outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                    print(f"CHECKPOINT SAVED: {outfile}")
                    rows_since_checkpoint = 0

                if rows_since_full_autosave >= full_autosave_every:
                    outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                    print(f"FULL AUTOSAVED: {outfile}")
                    rows_since_full_autosave = 0

                continue

        if relevance_prompt and not args.disable_llm_prefilter:
            relevance_text = relevance_prompt.format_map(payload)
            r_decision, r_conf, r_reason, r_raw = call_and_parse_relevance(
                relevance_model,
                relevance_text,
                anthropic_client,
                openai_client,
                gemini_client,
                retry_limit=retry_limit,
            )

            if r_decision == "DROP" and title_rescue:
                df.at[idx, "prefilter_decision"] = "KEEP"
                df.at[idx, "prefilter_confidence"] = r_conf
                df.at[idx, "prefilter_reason"] = (
                    "Rescued from DROP because canonical_name appears in thread_title. "
                    f"Original prefilter reason: {safe_text(r_reason)}"
                )
                df.at[idx, "prefilter_source"] = "llm_relevance_title_rescue"

                if args.debug_save_raw:
                    df.at[idx, "prefilter_raw"] = r_raw

                print(
                    f"[{counter}/{total_pending}] {event_key} | {row['canonical_name']} | "
                    f"source={row['source_bucket']} | PREFILTER_RESCUE_TITLE | "
                    f"reason={short_reason(r_reason)}"
                )

            else:
                df.at[idx, "prefilter_decision"] = r_decision
                df.at[idx, "prefilter_confidence"] = r_conf
                df.at[idx, "prefilter_reason"] = r_reason
                df.at[idx, "prefilter_source"] = "llm_relevance"

                if args.debug_save_raw:
                    df.at[idx, "prefilter_raw"] = r_raw

                if r_decision == "DROP":
                    df.at[idx, "final_reason"] = r_reason
                    df.at[idx, "final_reason_source"] = "prefilter_drop"

                    print(
                        f"[{counter}/{total_pending}] {event_key} | {row['canonical_name']} | "
                        f"source={row['source_bucket']} | PREFILTER_DROP | "
                        f"reason={short_reason(r_reason)}"
                    )

                    rows_since_checkpoint += 1
                    rows_since_full_autosave += 1

                    if rows_since_checkpoint >= checkpoint_every:
                        outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                        print(f"CHECKPOINT SAVED: {outfile}")
                        rows_since_checkpoint = 0

                    if rows_since_full_autosave >= full_autosave_every:
                        outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                        print(f"FULL AUTOSAVED: {outfile}")
                        rows_since_full_autosave = 0

                    continue
                    
        if firm_link_prompt:
            link_text = firm_link_prompt.format_map(payload)
            fl_label, fl_conf, fl_reason, fl_raw = call_and_parse_firm_link(
                link_text,
                anthropic_client,
                openai_client,
                gemini_client,
                model_name=firm_link_model,
                retry_limit=retry_limit,
            )

            df.at[idx, "firm_link_label"] = fl_label
            df.at[idx, "firm_link_confidence"] = fl_conf
            df.at[idx, "firm_link_reason"] = fl_reason

            if args.debug_save_raw:
                df.at[idx, "firm_link_raw"] = fl_raw

            if fl_label in {"ambient", "unrelated"}:
                df.at[idx, "final_sentiment"] = 0.0
                df.at[idx, "final_confidence"] = 0.0
                df.at[idx, "final_scum"] = 0.0
                df.at[idx, "final_reason"] = fl_reason
                df.at[idx, "final_reason_source"] = "firm_link_drop"

                print(
                    f"[{counter}/{total_pending}] {event_key} | {row['canonical_name']} | "
                    f"source={row['source_bucket']} | FIRM_LINK_DROP | "
                    f"label={fl_label} | conf={fl_conf} | reason={short_reason(fl_reason)}"
                )

                rows_since_checkpoint += 1
                rows_since_full_autosave += 1

                if rows_since_checkpoint >= checkpoint_every:
                    outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                    print(f"CHECKPOINT SAVED: {outfile}")
                    rows_since_checkpoint = 0

                if rows_since_full_autosave >= full_autosave_every:
                    outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                    print(f"FULL AUTOSAVED: {outfile}")
                    rows_since_full_autosave = 0

                continue

            if fl_label == "category":
                df.at[idx, "final_sentiment"] = 0.0
                df.at[idx, "final_confidence"] = 0.1
                df.at[idx, "final_scum"] = 0.0
                df.at[idx, "final_reason"] = fl_reason
                df.at[idx, "final_reason_source"] = "firm_link_category"

                print(
                    f"[{counter}/{total_pending}] {event_key} | {row['canonical_name']} | "
                    f"source={row['source_bucket']} | FIRM_LINK_CATEGORY | "
                    f"label={fl_label} | conf={fl_conf} | reason={short_reason(fl_reason)}"
                )

                rows_since_checkpoint += 1
                rows_since_full_autosave += 1

                if rows_since_checkpoint >= checkpoint_every:
                    outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                    print(f"CHECKPOINT SAVED: {outfile}")
                    rows_since_checkpoint = 0

                if rows_since_full_autosave >= full_autosave_every:
                    outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                    print(f"FULL AUTOSAVED: {outfile}")
                    rows_since_full_autosave = 0

                continue

        prompt = primary_prompt.format_map(payload)
        
        a_s, a_c, a_reason, a_raw, a_status = call_and_parse_sentiment(
            primary_a_model,
            prompt,
            anthropic_client,
            openai_client,
            gemini_client,
            retry_limit=retry_limit,
        )

        b_s, b_c, b_reason, b_raw, b_status = call_and_parse_sentiment(
            primary_b_model,
            prompt,
            anthropic_client,
            openai_client,
            gemini_client,
            retry_limit=retry_limit,
        )

        if args.debug_primary_a:
            print("\n--- PRIMARY_A RAW OUTPUT ---")
            print(a_raw)
            print("--- END PRIMARY_A RAW OUTPUT ---\n")

        if args.debug_primary_b:
            print("\n--- PRIMARY_B RAW OUTPUT ---")
            print(b_raw)
            print("--- END PRIMARY_B RAW OUTPUT ---\n")

        if a_status == "neutralized_abstention":
            print("\n--- PRIMARY_A ABSTENTION NORMALIZED TO NEUTRAL ---")
            print(a_raw)
            print("--- END PRIMARY_A ---\n")

        if b_status == "neutralized_abstention":
            print("\n--- PRIMARY_B ABSTENTION NORMALIZED TO NEUTRAL ---")
            print(b_raw)
            print("--- END PRIMARY_B ---\n")
        
        sig = df.at[idx, "signal_strength"] or 0
        scale = 0.5 + 0.5 * (sig / 4)  # 0.5 → 1.0
        
        adj_a_c = a_c * scale if a_c is not None else None
        adj_b_c = b_c * scale if b_c is not None else None
        
        a_sc = scum_score(a_s, adj_a_c)
        b_sc = scum_score(b_s, adj_b_c)

        df.at[idx, "primary_a_sentiment"] = a_s
        df.at[idx, "primary_a_confidence"] = adj_a_c
        df.at[idx, "primary_a_scum"] = a_sc
        df.at[idx, "primary_a_reason"] = a_reason
        df.at[idx, "primary_a_status"] = a_status

        df.at[idx, "primary_b_sentiment"] = b_s
        df.at[idx, "primary_b_confidence"] = adj_b_c
        df.at[idx, "primary_b_scum"] = b_sc
        df.at[idx, "primary_b_reason"] = b_reason
        df.at[idx, "primary_b_status"] = b_status

        if args.debug_save_raw:
            df.at[idx, "primary_a_raw"] = a_raw
            df.at[idx, "primary_b_raw"] = b_raw
        
        a_valid = is_valid_score(a_s, adj_a_c)
        b_valid = is_valid_score(b_s, adj_b_c)
        
        if not a_valid and not b_valid:
            df.at[idx, "solomon_triggered"] = False
            df.at[idx, "final_sentiment"] = None
            df.at[idx, "final_confidence"] = None
            df.at[idx, "final_scum"] = None
            df.at[idx, "final_reason"] = (
                f"Primary failure. A_status={a_status}, B_status={b_status}. "
                f"A_reason={safe_text(a_reason)} | B_reason={safe_text(b_reason)}"
            )
            df.at[idx, "final_reason_source"] = "primary_failure"
        
            print(
                f"[{counter}/{total_pending}] {event_key} | {row['canonical_name']} | "
                f"source={row['source_bucket']} | PRIMARY FAILURE | "
                f"A_status={a_status} | B_status={b_status}"
            )
        
            rows_since_checkpoint += 1
            rows_since_full_autosave += 1
        
            if rows_since_checkpoint >= checkpoint_every:
                outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                print(f"CHECKPOINT SAVED: {outfile}")
                rows_since_checkpoint = 0
        
            if rows_since_full_autosave >= full_autosave_every:
                outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                print(f"FULL AUTOSAVED: {outfile}")
                rows_since_full_autosave = 0
        
            continue
        
        if a_valid and not b_valid:
            final_s = round(a_s, 4)
            final_c = round(adj_a_c, 4) if adj_a_c is not None else round(a_c, 4)
            final_sc = a_sc
            final_reason = f"A_only:{safe_text(a_reason)} | B_failed:{b_status}"
            final_reason_source = "primary_a_only"
        
            df.at[idx, "solomon_triggered"] = False
            df.at[idx, "final_sentiment"] = final_s
            df.at[idx, "final_confidence"] = final_c
            df.at[idx, "final_scum"] = final_sc
            df.at[idx, "final_reason"] = final_reason
            df.at[idx, "final_reason_source"] = final_reason_source
        
            print(
                f"[{counter}/{total_pending}] "
                f"{event_key} | {row['canonical_name']} | source={row['source_bucket']} | "
                f"A={a_sc} | B={b_sc} | solomon=False | final={final_sc} | "
                f"final_source={final_reason_source}"
            )
        
            rows_since_checkpoint += 1
            rows_since_full_autosave += 1
        
            if rows_since_checkpoint >= checkpoint_every:
                outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                print(f"CHECKPOINT SAVED: {outfile}")
                rows_since_checkpoint = 0
        
            if rows_since_full_autosave >= full_autosave_every:
                outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                print(f"FULL AUTOSAVED: {outfile}")
                rows_since_full_autosave = 0
        
            continue
        
        if b_valid and not a_valid:
            final_s = round(b_s, 4)
            final_c = round(adj_b_c, 4) if adj_b_c is not None else round(b_c, 4)
            final_sc = b_sc
            final_reason = f"B_only:{safe_text(b_reason)} | A_failed:{a_status}"
            final_reason_source = "primary_b_only"
        
            df.at[idx, "solomon_triggered"] = False
            df.at[idx, "final_sentiment"] = final_s
            df.at[idx, "final_confidence"] = final_c
            df.at[idx, "final_scum"] = final_sc
            df.at[idx, "final_reason"] = final_reason
            df.at[idx, "final_reason_source"] = final_reason_source
        
            print(
                f"[{counter}/{total_pending}] "
                f"{event_key} | {row['canonical_name']} | source={row['source_bucket']} | "
                f"A={a_sc} | B={b_sc} | solomon=False | final={final_sc} | "
                f"final_source={final_reason_source}"
            )
        
            rows_since_checkpoint += 1
            rows_since_full_autosave += 1
        
            if rows_since_checkpoint >= checkpoint_every:
                outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                print(f"CHECKPOINT SAVED: {outfile}")
                rows_since_checkpoint = 0
        
            if rows_since_full_autosave >= full_autosave_every:
                outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
                print(f"FULL AUTOSAVED: {outfile}")
                rows_since_full_autosave = 0
        
            continue
            
        solomon = should_trigger_solomon(a_s, adj_a_c, a_sc, b_s, adj_b_c, b_sc, disagree_threshold)
        df.at[idx, "solomon_triggered"] = solomon

        if solomon:
            tie_payload = build_format_payload(
                row,
                model_a_sentiment=a_s,
                model_a_confidence=a_c,
                model_b_sentiment=b_s,
                model_b_confidence=b_c,
            )
            t_prompt = tie_prompt.replace("{", "{{").replace("}", "}}")
            for k in tie_payload:
                t_prompt = t_prompt.replace("{{" + k + "}}", "{" + k + "}")
            t_prompt = t_prompt.format_map(tie_payload)

            t_s, t_c, t_reason, t_raw, t_status = call_and_parse_sentiment(
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
            df.at[idx, "tiebreaker_status"] = t_status

            if args.debug_tiebreaker:
                print("\n--- TIEBREAKER RAW OUTPUT ---")
                print(t_raw)
                print("--- END TIEBREAKER RAW OUTPUT ---\n")

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
        else:
            final_s = round((a_s + b_s) / 2, 4)
            final_c = round((a_c + b_c) / 2, 4)
            final_sc = scum_score(final_s, final_c)
            final_reason = f"A:{safe_text(a_reason)} | B:{safe_text(b_reason)}"
            final_reason_source = "primary_avg"

        df.at[idx, "final_sentiment"] = final_s
        df.at[idx, "final_confidence"] = final_c
        df.at[idx, "final_scum"] = final_sc
        df.at[idx, "final_reason"] = final_reason
        df.at[idx, "final_reason_source"] = final_reason_source

        print(
            f"[{counter}/{total_pending}] "
            f"{event_key} | {row['canonical_name']} | source={row['source_bucket']} | "
            f"A={a_sc} (c={adj_a_c}) | B={b_sc} (c={adj_b_c}) | solomon={solomon} | final={final_sc} | "
            f"final_source={final_reason_source} | "
            f"A_reason={short_reason(a_reason)} | "
            f"B_reason={short_reason(b_reason)}"
        )

        rows_since_checkpoint += 1
        rows_since_full_autosave += 1

        if rows_since_checkpoint >= checkpoint_every:
            outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
            print(f"CHECKPOINT SAVED: {outfile}")
            rows_since_checkpoint = 0

        if rows_since_full_autosave >= full_autosave_every:
            outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
            print(f"FULL AUTOSAVED: {outfile}")
            rows_since_full_autosave = 0

    checkpoint_outfile = save_checkpoint_df(df, paths["scored_events_dir"], prefix=args.output_prefix)
    latest_outfile = save_df_clean(df, paths["scored_events_dir"], prefix=args.output_prefix)
    final_outfile = save_final_df(df, paths["scored_events_dir"], prefix=args.output_prefix)

    print("\nLatest checkpoint save:", checkpoint_outfile)
    print("Latest rolling save:", latest_outfile)
    print("Saved final:", final_outfile)
    print("\nFinal scored rows:", int(df["final_scum"].notna().sum()))
    print("\nFinal reason sources:")
    print(df["final_reason_source"].value_counts(dropna=False))

    if "prefilter_decision" in df.columns:
        print("\nPrefilter decisions:")
        print(df["prefilter_decision"].value_counts(dropna=False))

    if "title_inheritance_rescue" in df.columns:
        print("\nTitle inheritance rescues:")
        rescue_series = df["title_inheritance_rescue"].astype("boolean")
        print(rescue_series.value_counts(dropna=False))

    print("\nPrimary validity:")
    print("A valid:", int(pd.Series(df["primary_a_status"]).isin(["valid", "neutralized_abstention"]).sum()))
    print("B valid:", int(pd.Series(df["primary_b_status"]).isin(["valid", "neutralized_abstention"]).sum()))

    print("\nPrimary statuses:")
    print("A:")
    print(pd.Series(df["primary_a_status"]).value_counts(dropna=False))
    print("\nB:")
    print(pd.Series(df["primary_b_status"]).value_counts(dropna=False))


if __name__ == "__main__":
    main()
