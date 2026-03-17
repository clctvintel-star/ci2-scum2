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
# small helper
# ==========================================================

def short_reason(txt, n=80):
    if not txt:
        return ""
    txt = txt.replace("\n", " ")
    return txt[:n] + ("..." if len(txt) > n else "")


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
        default=None
    )

    parser.add_argument(
        "--sample-per-firm",
        type=int,
        default=None
    )

    parser.add_argument("--event-file", type=str, default=None)
    parser.add_argument("--thread-file", type=str, default=None)

    parser.add_argument(
        "--show-latest-files",
        action="store_true"
    )

    parser.add_argument(
        "--debug-primary-b",
        action="store_true"
    )

    parser.add_argument(
        "--debug-save-raw",
        action="store_true"
    )

    return parser.parse_args()


# ==========================================================
# helpers
# ==========================================================

class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def latest_parquet(directory: str, prefix: Optional[str] = None) -> Path:
    pattern = "*.parquet" if prefix is None else f"{prefix}_*.parquet"
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
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


def autosave_df(df, outdir):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outfile = Path(outdir) / f"scored_events_{ts}.parquet"
    df.to_parquet(outfile, index=False)
    return outfile


def scum_score(sentiment, confidence):
    if sentiment is None or confidence is None:
        return None
    return round(float(sentiment) * math.sqrt(float(confidence)), 4)


def clamp_float(v, lo, hi):
    if v is None:
        return None
    try:
        v = float(v)
    except:
        return None
    return max(lo, min(hi, v))


def safe_text(v):
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def normalize_source_bucket(candidate_source):
    txt = safe_text(candidate_source).lower()
    if txt == "direct_event_match":
        return "direct"
    if txt == "thread_context":
        return "context"
    return "unknown"


# ==========================================================
# parsing
# ==========================================================

def parse_relevance_response(text):

    decision=None
    conf=None
    reason=None

    m=re.search(r"Decision:\s*(KEEP|DROP|UNCERTAIN)",text,re.I)
    if m:
        decision=m.group(1).upper()

    m=re.search(r"Confidence:\s*([0-9.]+)",text,re.I)
    if m:
        conf=float(m.group(1))

    m=re.search(r"Reason:\s*(.+)",text,re.I|re.S)
    if m:
        reason=m.group(1).strip()

    conf=clamp_float(conf,0,1)

    return decision,conf,reason


def extract_json_payload(text):

    if not isinstance(text,str):
        return None

    text=text.strip()

    try:
        return json.loads(text)
    except:
        pass

    m=re.search(r"\{.*\}",text,re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass

    return None


def parse_sentiment_response(text):

    payload=extract_json_payload(text)
    if not payload:
        return None,None,None

    sentiment=payload.get("sentiment")
    confidence=payload.get("confidence")
    explanation=payload.get("explanation")

    sentiment=clamp_float(sentiment,-1,1)
    confidence=clamp_float(confidence,0,1)

    return sentiment,confidence,explanation


def is_missing_score(s,c):
    return s is None or c is None


def should_trigger_solomon(a_s,a_c,a_sc,b_s,b_c,b_sc,thresh):

    if is_missing_score(a_s,a_c) or is_missing_score(b_s,b_c):
        return True

    if a_s*b_s<0:
        return True

    if a_sc is None or b_sc is None:
        return True

    if abs(a_sc-b_sc)>thresh:
        return True

    return False


# ==========================================================
# clients
# ==========================================================

def build_clients(env):

    anthropic_client=Anthropic(api_key=env["ANTHROPIC_API_KEY"])
    openai_client=OpenAI(api_key=env["OPENAI_API_KEY"])
    gemini_client=genai.Client(api_key=env["GOOGLE_API_KEY"])

    return anthropic_client,openai_client,gemini_client


# ==========================================================
# model calls
# ==========================================================

def call_model(model_name,prompt,anthropic_client,openai_client,gemini_client,max_tokens=250,temperature=0.1):

    if model_name.startswith("claude"):
        r=anthropic_client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role":"user","content":prompt}]
        )
        return r.content[0].text

    if model_name.startswith("gpt"):
        r=openai_client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role":"user","content":prompt}]
        )
        return r.choices[0].message.content

    if model_name.startswith("gemini"):
        r=gemini_client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return r.text

    raise ValueError(model_name)


def call_and_parse_sentiment(model,prompt,anthropic_client,openai_client,gemini_client):

    raw=call_model(model,prompt,anthropic_client,openai_client,gemini_client)

    sentiment,confidence,reason=parse_sentiment_response(raw)

    return sentiment,confidence,reason,raw


# ==========================================================
# dataframe setup
# ==========================================================

def prepare_scoring_df(event_df,thread_df):

    scored_df=event_df.copy()

    scored_df["source_bucket"]=scored_df["candidate_source"].apply(normalize_source_bucket)

    out_cols=[

        "relevance_decision",
        "relevance_confidence",
        "relevance_reason",

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

        "solomon_triggered",

        "primary_a_raw",
        "primary_b_raw",
        "tiebreaker_raw"
    ]

    for c in out_cols:
        if c not in scored_df.columns:
            scored_df[c]=None

    return scored_df


# ==========================================================
# main
# ==========================================================

def main():

    args=parse_args()

    _,paths,settings=load_configs()
    env=load_env(paths["env_path"])

    ensure_dir(paths["scored_events_dir"])

    anthropic_client,openai_client,gemini_client=build_clients(env)

    primary_prompt=read_text(Path("prompts/scum_primary_prompt.txt"))
    tie_prompt=read_text(Path("prompts/scum_tiebreaker_prompt.txt"))

    event_file=resolve_input_file(args.event_file,paths["event_firm_dir"],"event_firm_pairs")
    thread_file=resolve_input_file(args.thread_file,paths["thread_firm_dir"],"thread_firm_pairs")

    event_df=pd.read_parquet(event_file)
    thread_df=pd.read_parquet(thread_file)

    df=prepare_scoring_df(event_df,thread_df)

    print("Rows queued:",len(df))

    for idx,row in df.iterrows():

        payload=SafeDict({
            "FIRM_NAME":row["canonical_name"],
            "EVENT_TEXT":safe_text(row.get("event_text")),
            "THREAD_TEXT":safe_text(row.get("thread_text"))
        })

        prompt=primary_prompt.format_map(payload)

        a_s,a_c,a_reason,a_raw=call_and_parse_sentiment(
            settings["models"]["score"]["primary_a"],
            prompt,
            anthropic_client,
            openai_client,
            gemini_client
        )

        b_s,b_c,b_reason,b_raw=call_and_parse_sentiment(
            settings["models"]["score"]["primary_b"],
            prompt,
            anthropic_client,
            openai_client,
            gemini_client
        )

        a_sc=scum_score(a_s,a_c)
        b_sc=scum_score(b_s,b_c)

        df.at[idx,"primary_a_sentiment"]=a_s
        df.at[idx,"primary_a_confidence"]=a_c
        df.at[idx,"primary_a_scum"]=a_sc
        df.at[idx,"primary_a_reason"]=a_reason

        df.at[idx,"primary_b_sentiment"]=b_s
        df.at[idx,"primary_b_confidence"]=b_c
        df.at[idx,"primary_b_scum"]=b_sc
        df.at[idx,"primary_b_reason"]=b_reason

        solomon=should_trigger_solomon(a_s,a_c,a_sc,b_s,b_c,b_sc,0.2)

        final_sc=None
        final_s=None
        final_c=None

        if solomon:

            tie=SafeDict({
                **payload,
                "MODEL_A_SENTIMENT":a_s,
                "MODEL_A_CONFIDENCE":a_c,
                "MODEL_B_SENTIMENT":b_s,
                "MODEL_B_CONFIDENCE":b_c
            })

            t_prompt=tie_prompt.format_map(tie)

            t_s,t_c,t_reason,t_raw=call_and_parse_sentiment(
                settings["models"]["score"]["tiebreaker"],
                t_prompt,
                anthropic_client,
                openai_client,
                gemini_client
            )

            t_sc=scum_score(t_s,t_c)

            df.at[idx,"tiebreaker_sentiment"]=t_s
            df.at[idx,"tiebreaker_confidence"]=t_c
            df.at[idx,"tiebreaker_scum"]=t_sc
            df.at[idx,"tiebreaker_reason"]=t_reason

            final_s=t_s
            final_c=t_c
            final_sc=t_sc

        else:

            final_s=(a_s+b_s)/2
            final_c=(a_c+b_c)/2
            final_sc=scum_score(final_s,final_c)

        df.at[idx,"final_sentiment"]=final_s
        df.at[idx,"final_confidence"]=final_c
        df.at[idx,"final_scum"]=final_sc

        print(
            f"{row['event_id']} | {row['canonical_name']} | source={row['source_bucket']} | "
            f"A={a_sc} | B={b_sc} | solomon={solomon} | final={final_sc} | "
            f"A_reason={short_reason(a_reason)} | "
            f"B_reason={short_reason(b_reason)}"
        )

    outfile=autosave_df(df,paths["scored_events_dir"])

    print("\nSaved:",outfile)


if __name__=="__main__":
    main()
