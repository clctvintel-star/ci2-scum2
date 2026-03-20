import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from anthropic import Anthropic

from utils import load_configs, load_env, ensure_dir, load_prompt


# =========================================================
# RUNTIME CONTROLS
# =========================================================
# "full" = rebuild thread_firm_pairs with LLM + rebuild event_firm_pairs
# "rebuild_events_from_existing_threads" = skip LLM, load existing thread_firm_pairs, rebuild only event_firm_pairs
RUN_MODE = "full"

# If RUN_MODE == "rebuild_events_from_existing_threads":
# - if this is None, the script will auto-pick the latest thread_firm_pairs_*.parquet
# - otherwise it will use the exact path you provide
EXISTING_THREAD_FIRM_PAIRS_PATH = None

# Autosave cadence
THREAD_AUTOSAVE_EVERY = 100
EVENT_AUTOSAVE_EVERY = 50000

# LLM retry settings
LLM_RETRY_LIMIT = 3
LLM_RETRY_SLEEP_SECONDS = 1

# If True, forces lowercase mention_type normalization and strips weird variants
NORMALIZE_MENTION_TYPES = True

# If True, risky-alias regex hits require LLM confirmation at thread level
REQUIRE_LLM_CONFIRMATION_FOR_RISKY_REGEX_ONLY = True


# =========================================================
# CONSTANTS
# =========================================================
THREAD_FIRM_REQUIRED_COLS = [
    "post_id",
    "canonical_name",
    "matched_aliases",
    "thread_direct_match",
    "thread_text",
    "thread_title",
    "subreddit",
    "thread_created_utc",
    "candidate_strength",
    "mention_type",
    "llm_rationale",
    "regex_candidate_names",
    "llm_candidate_names",
    "llm_called",
]

EVENT_FIRM_REQUIRED_COLS = [
    "event_id",
    "post_id",
    "comment_id",
    "parent_id",
    "parent_text",
    "canonical_name",
    "matched_aliases",
    "candidate_source",
    "event_has_direct_regex_match",
    "event_text",
    "created_utc",
    "depth",
    "score",
    "subreddit",
]

THREAD_FIRM_EMPTY_DF = pd.DataFrame(columns=THREAD_FIRM_REQUIRED_COLS)
EVENT_FIRM_EMPTY_DF = pd.DataFrame(columns=EVENT_FIRM_REQUIRED_COLS)


# =========================================================
# HELPERS
# =========================================================
def find_latest_parquet(path_obj: Path, prefix: str) -> Path:
    files = sorted(path_obj.glob(f"{prefix}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No files found in {path_obj} matching {prefix}_*.parquet")
    return max(files, key=lambda p: p.stat().st_mtime)


def normalize_mention_type(value: Optional[str]) -> str:
    if value is None:
        return "none"

    txt = str(value).strip().lower()

    if txt in {"", "none", "null", "n/a", "na"}:
        return "none"
    if txt in {"explicit", "direct", "directly"}:
        return "explicit"
    if txt in {"indirect", "contextual", "context"}:
        return "indirect"
    if txt in {"ambiguous", "unclear", "maybe"}:
        return "ambiguous"

    return "none"


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def safe_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_alias_patterns(firms_config) -> List[Dict]:
    patterns = []

    for firm in firms_config["firms"]:
        canonical_name = firm["canonical_name"]
        aliases = firm.get("aliases", [])
        risky_aliases = set(a.lower() for a in firm.get("risky_aliases", []))

        compiled = []
        for alias in aliases:
            alias_clean = safe_str(alias)
            if not alias_clean:
                continue

            is_risky = alias_clean.lower() in risky_aliases

            if is_risky or len(alias_clean) <= 3:
                pattern = re.compile(
                    rf"(?<!\w){re.escape(alias_clean)}(?!\w)",
                    flags=re.IGNORECASE,
                )
            else:
                pattern = re.compile(
                    rf"\b{re.escape(alias_clean)}\b",
                    flags=re.IGNORECASE,
                )

            compiled.append((alias_clean, pattern, is_risky))

        patterns.append({
            "canonical_name": canonical_name,
            "patterns": compiled,
        })

    return patterns


def build_alias_lookup(alias_patterns: List[Dict]) -> Dict[str, List[Tuple[str, re.Pattern, bool]]]:
    return {
        x["canonical_name"]: x["patterns"]
        for x in alias_patterns
    }
    
def build_parent_lookup(events_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Maps:
      t1_<comment_id> -> {"text": comment_text, "parent_id": parent_id}
      t3_<post_id>    -> {"text": post_text,    "parent_id": None}

    Important:
    - post text must come ONLY from the actual post row
    - comment rows must NOT overwrite post text
    """
    lookup: Dict[str, Dict] = {}

    for _, row in events_df.iterrows():
        event_text = row.get("event_text", "")
        parent_id = row.get("parent_id")
        event_type = row.get("type")

        comment_id = row.get("comment_id")
        post_id = row.get("post_id")

        if event_type == "comment" and pd.notna(comment_id):
            lookup[f"t1_{comment_id}"] = {
                "text": event_text,
                "parent_id": parent_id,
            }

        elif event_type == "post" and pd.notna(post_id):
            lookup[f"t3_{post_id}"] = {
                "text": event_text,
                "parent_id": None,
            }

    return lookup


def load_latest_events(events_dir: str) -> pd.DataFrame:
    files = sorted(Path(events_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {events_dir}")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["event_id"]).copy()
    return df


def assemble_event_text(row) -> str:
    parts = []
    for col in ["title", "selftext", "body"]:
        val = row.get(col)
        if pd.notna(val) and val:
            parts.append(str(val))
    return "\n\n".join(parts).strip()


def build_thread_text(group: pd.DataFrame) -> str:
    texts = []
    group = group.sort_values(["created_utc", "depth"], ascending=[True, True])

    for _, row in group.iterrows():
        txt = row["event_text"]
        if isinstance(txt, str) and txt.strip():
            texts.append(txt)

    return "\n\n".join(texts).strip()


def regex_matches(text: str, alias_patterns: List[Dict]) -> List[Dict]:
    matched = []

    if not isinstance(text, str) or not text.strip():
        return matched

    for firm in alias_patterns:
        all_aliases = []
        risky_aliases = []
        safe_aliases = []

        for alias, pattern, is_risky in firm["patterns"]:
            if pattern.search(text):
                all_aliases.append(alias)
                if is_risky:
                    risky_aliases.append(alias)
                else:
                    safe_aliases.append(alias)

        if all_aliases:
            matched.append({
                "canonical_name": firm["canonical_name"],
                "matched_aliases": sorted(set(all_aliases)),
                "matched_safe_aliases": sorted(set(safe_aliases)),
                "matched_risky_aliases": sorted(set(risky_aliases)),
                "has_non_risky_match": len(safe_aliases) > 0,
                "has_risky_match": len(risky_aliases) > 0,
            })

    return matched


def regex_matches_for_firm(text: str, firm_patterns: List[Tuple[str, re.Pattern, bool]]) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    hits = []
    for alias, pattern, _ in firm_patterns:
        if pattern.search(text):
            hits.append(alias)

    return sorted(set(hits))


def safe_json_load(text: str):
    text = str(text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"(\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return []

    return []


def call_thread_detector(client, model_name: str, prompt: str):
    for attempt in range(1, LLM_RETRY_LIMIT + 1):
        try:
            resp = client.messages.create(
                model=model_name,
                max_tokens=1200,
                temperature=0,
                system="Return JSON only.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            return safe_json_load(text)

        except Exception as e:
            print(f"LLM thread detector error (attempt {attempt}/{LLM_RETRY_LIMIT}), retrying: {e}")
            time.sleep(LLM_RETRY_SLEEP_SECONDS)

    return []


def build_detector_prompt(
    prompt_template: str,
    canonical_names: List[str],
    thread_title: str,
    thread_text: str,
    regex_candidates: List[str],
) -> str:
    return prompt_template.format(
        FIRM_UNIVERSE=json.dumps(canonical_names, ensure_ascii=False),
        REGEX_CANDIDATES=json.dumps(regex_candidates, ensure_ascii=False),
        THREAD_TITLE=thread_title or "",
        THREAD_TEXT=thread_text or "",
    )


def normalize_llm_candidates(llm_output, allowed_names: List[str]) -> List[Dict]:
    rows = []

    if not isinstance(llm_output, list):
        return rows

    allowed_set = set(allowed_names)

    for item in llm_output:
        if not isinstance(item, dict):
            continue

        canonical_name = item.get("canonical_name")
        if canonical_name not in allowed_set:
            continue

        mention_type = normalize_mention_type(item.get("mention_type")) if NORMALIZE_MENTION_TYPES else item.get("mention_type")
        confidence = safe_float(item.get("confidence"))
        rationale = item.get("rationale")

        rows.append({
            "canonical_name": canonical_name,
            "mention_type": mention_type,
            "llm_confidence": confidence,
            "llm_rationale": rationale,
        })

    if not rows:
        return rows

    tmp = pd.DataFrame(rows)
    tmp["llm_confidence_sort"] = tmp["llm_confidence"].fillna(-1.0)
    tmp = (
        tmp.sort_values(["canonical_name", "llm_confidence_sort"], ascending=[True, False])
        .drop_duplicates(subset=["canonical_name"], keep="first")
        .drop(columns=["llm_confidence_sort"])
    )
    return tmp.to_dict(orient="records")


def validate_thread_firm_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return THREAD_FIRM_EMPTY_DF.copy()

    for col in THREAD_FIRM_REQUIRED_COLS:
        if col not in df.columns:
            df[col] = None

    if NORMALIZE_MENTION_TYPES and "mention_type" in df.columns:
        df["mention_type"] = df["mention_type"].apply(normalize_mention_type)

    df = df[THREAD_FIRM_REQUIRED_COLS].copy()
    df = df.drop_duplicates(subset=["post_id", "canonical_name"]).copy()
    return df


def validate_event_firm_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return EVENT_FIRM_EMPTY_DF.copy()

    for col in EVENT_FIRM_REQUIRED_COLS:
        if col not in df.columns:
            df[col] = None

    df = df[EVENT_FIRM_REQUIRED_COLS].copy()
    df = df.drop_duplicates(subset=["event_id", "canonical_name"]).copy()
    return df


def should_keep_thread_candidate(
    canonical_name: str,
    regex_info: Optional[Dict],
    llm_info: Optional[Dict],
) -> bool:
    llm_confirmed = llm_info is not None
    regex_present = regex_info is not None

    if llm_confirmed:
        return True

    if not regex_present:
        return False

    has_non_risky_match = bool(regex_info.get("has_non_risky_match", False))

    if has_non_risky_match:
        return True

    if REQUIRE_LLM_CONFIRMATION_FOR_RISKY_REGEX_ONLY:
        return False

    return True

def build_ancestry_text(parent_id: str, parent_lookup: Dict[str, Dict], max_depth: int = 3) -> str:
    texts = []
    current_id = parent_id
    seen = set()

    for _ in range(max_depth):
        if not current_id or current_id in seen:
            break

        seen.add(current_id)

        node = parent_lookup.get(current_id)
        if not node:
            break

        text = node.get("text", "")
        if not text:
            break

        texts.append(text)

        current_id = node.get("parent_id")

    texts = list(dict.fromkeys(texts))
    texts.reverse()

    return "\n\n".join(texts)
    
# =========================================================
# THREAD-LEVEL BUILD
# =========================================================
def build_thread_firm_pairs(
    events_df: pd.DataFrame,
    alias_patterns: List[Dict],
    detector_client,
    detector_model: str,
    detector_prompt: str,
    thread_firm_dir: str,
    autosave_every: int = 100,
) -> pd.DataFrame:
    rows = []
    grouped_items = list(events_df.groupby("post_id", dropna=True))
    total_threads = len(grouped_items)
    canonical_names = [f["canonical_name"] for f in alias_patterns]

    autosave_path = Path(thread_firm_dir) / "thread_firm_autosave.parquet"

    for i, (post_id, group) in enumerate(grouped_items, start=1):
        post_rows = group[group["type"] == "post"].copy()
        if post_rows.empty:
            print(f"[{i}/{total_threads}] thread {post_id}: skipped (no post row)")
            continue

        post_row = post_rows.iloc[0]
        thread_title = post_row.get("title")
        thread_created_utc = post_row.get("created_utc")
        subreddit = post_row.get("subreddit")
        thread_text = build_thread_text(group)

        regex_hits = regex_matches(thread_text, alias_patterns)
        regex_alias_map = {x["canonical_name"]: x for x in regex_hits}
        regex_candidate_names = sorted(regex_alias_map.keys())

        prompt = build_detector_prompt(
            detector_prompt,
            canonical_names=canonical_names,
            thread_title=thread_title,
            thread_text=thread_text,
            regex_candidates=regex_candidate_names,
        )

        llm_output = call_thread_detector(
            client=detector_client,
            model_name=detector_model,
            prompt=prompt,
        )
        llm_hits = normalize_llm_candidates(llm_output, canonical_names)
        llm_map = {x["canonical_name"]: x for x in llm_hits}
        llm_candidate_names = sorted(llm_map.keys())

        candidate_names_to_consider = sorted(set(regex_candidate_names) | set(llm_candidate_names))
        final_names = []

        for canonical_name in candidate_names_to_consider:
            regex_info = regex_alias_map.get(canonical_name)
            llm_info = llm_map.get(canonical_name)

            if should_keep_thread_candidate(
                canonical_name=canonical_name,
                regex_info=regex_info,
                llm_info=llm_info,
            ):
                final_names.append(canonical_name)

        for canonical_name in final_names:
            regex_info = regex_alias_map.get(canonical_name, {})
            llm_info = llm_map.get(canonical_name, {})

            matched_aliases = regex_info.get("matched_aliases", [])
            thread_direct_match = bool(regex_info.get("has_non_risky_match", False) or llm_info)

            rows.append({
                "post_id": post_id,
                "canonical_name": canonical_name,
                "matched_aliases": ", ".join(matched_aliases) if matched_aliases else None,
                "thread_direct_match": thread_direct_match,
                "thread_text": thread_text,
                "thread_title": thread_title,
                "subreddit": subreddit,
                "thread_created_utc": thread_created_utc,
                "candidate_strength": llm_info.get("llm_confidence"),
                "mention_type": normalize_mention_type(llm_info.get("mention_type")),
                "llm_rationale": llm_info.get("llm_rationale"),
                "regex_candidate_names": ", ".join(regex_candidate_names) if regex_candidate_names else None,
                "llm_candidate_names": ", ".join(llm_candidate_names) if llm_candidate_names else None,
                "llm_called": True,
            })

        print(
            f"[{i}/{total_threads}] thread {post_id}: "
            f"regex={len(regex_hits)} llm={len(llm_hits)} final={len(final_names)}"
        )

        if i % autosave_every == 0:
            tmp_df = validate_thread_firm_df(pd.DataFrame(rows))
            tmp_df.to_parquet(autosave_path, index=False)
            print(f"AUTOSAVED THREADS: {len(tmp_df)} rows -> {autosave_path}")

    final_df = validate_thread_firm_df(pd.DataFrame(rows))
    final_df.to_parquet(autosave_path, index=False)
    print(f"FINAL THREAD AUTOSAVE: {len(final_df)} rows -> {autosave_path}")

    return final_df


def load_existing_thread_firm_pairs(thread_firm_dir: str, explicit_path: Optional[str]) -> pd.DataFrame:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"EXISTING_THREAD_FIRM_PAIRS_PATH not found: {path}")
    else:
        path = find_latest_parquet(Path(thread_firm_dir), "thread_firm_pairs")

    print(f"Loading existing thread_firm_pairs from: {path}")
    df = pd.read_parquet(path).copy()
    df = validate_thread_firm_df(df)
    print(f"Loaded existing thread_firm_pairs rows: {len(df)}")
    return df


# =========================================================
# EVENT-LEVEL BUILD
# =========================================================
def build_thread_firm_map(thread_firm_df: pd.DataFrame) -> Dict[str, List[Dict]]:
    thread_map: Dict[str, List[Dict]] = {}

    for _, row in thread_firm_df.iterrows():
        thread_map.setdefault(row["post_id"], []).append({
            "canonical_name": row["canonical_name"],
            "thread_matched_aliases": row.get("matched_aliases"),
            "thread_direct_match": bool(row.get("thread_direct_match", False)),
        })

    return thread_map


def build_event_firm_pairs(
    events_df: pd.DataFrame,
    thread_firm_df: pd.DataFrame,
    alias_patterns: List[Dict],
    event_firm_dir: str,
    autosave_every: int = 50000,
) -> pd.DataFrame:
    parent_lookup = build_parent_lookup(events_df)

    rows = []
    alias_lookup = build_alias_lookup(alias_patterns)
    thread_firm_map = build_thread_firm_map(thread_firm_df)

    autosave_path = Path(event_firm_dir) / "event_firm_autosave.parquet"
    total_events = len(events_df)

    for i, (_, row) in enumerate(events_df.iterrows(), start=1):
        raw_event_text = row["event_text"]
        post_id = row["post_id"]
        parent_id = row["parent_id"]

        parent_text = build_ancestry_text(parent_id, parent_lookup, max_depth=3)
        thread_firms = thread_firm_map.get(post_id, [])

        if not thread_firms:
            if i % 25000 == 0 or i == total_events:
                print(f"[event {i}/{total_events}] built_rows={len(rows)}")
            continue

        for thread_firm in thread_firms:
            canonical_name = thread_firm["canonical_name"]
            firm_patterns = alias_lookup.get(canonical_name, [])

            # direct match should be checked on the child event itself, not the stitched ancestry blob
            direct_aliases = regex_matches_for_firm(raw_event_text, firm_patterns)

            if direct_aliases:
                candidate_source = "direct_event_match"
                matched_aliases = ", ".join(direct_aliases)
                event_has_direct_regex_match = True
            else:
                candidate_source = "thread_context"
                matched_aliases = None
                event_has_direct_regex_match = False

            rows.append({
                "event_id": row["event_id"],
                "post_id": post_id,
                "comment_id": row["comment_id"],
                "parent_id": parent_id,
                "parent_text": parent_text if parent_text else None,
                "canonical_name": canonical_name,
                "matched_aliases": matched_aliases,
                "candidate_source": candidate_source,
                "event_has_direct_regex_match": event_has_direct_regex_match,
                "event_text": raw_event_text,
                "created_utc": row["created_utc"],
                "depth": row["depth"],
                "score": row["score"],
                "subreddit": row["subreddit"],
            })

        if i % autosave_every == 0:
            tmp_df = validate_event_firm_df(pd.DataFrame(rows))
            tmp_df.to_parquet(autosave_path, index=False)
            print(f"AUTOSAVED EVENTS: {len(tmp_df)} rows -> {autosave_path}")

        if i % 25000 == 0 or i == total_events:
            print(f"[event {i}/{total_events}] built_rows={len(rows)}")

    final_df = validate_event_firm_df(pd.DataFrame(rows))
    final_df.to_parquet(autosave_path, index=False)
    print(f"FINAL EVENT AUTOSAVE: {len(final_df)} rows -> {autosave_path}")

    return final_df


# =========================================================
# MAIN
# =========================================================
def main():
    firms, paths, settings = load_configs()
    env = load_env(paths["env_path"])

    ensure_dir(paths["thread_firm_dir"])
    ensure_dir(paths["event_firm_dir"])

    alias_patterns = build_alias_patterns(firms)

    events_df = load_latest_events(paths["events_dir"]).copy()
    events_df["event_text"] = events_df.apply(assemble_event_text, axis=1)

    print(f"Loaded raw events: {len(events_df)}")
    print(f"Unique threads: {events_df['post_id'].nunique()}")

    if RUN_MODE == "full":
        detector_model = settings["models"]["expand"]["thread_detector"]
        detector_prompt = load_prompt(
            str(Path(__file__).resolve().parents[1] / "prompts" / "relevance_prompt.txt")
        )

        if not env["ANTHROPIC_API_KEY"]:
            raise ValueError("ANTHROPIC_API_KEY missing from env")

        detector_client = Anthropic(api_key=env["ANTHROPIC_API_KEY"])

        thread_firm_df = build_thread_firm_pairs(
            events_df=events_df,
            alias_patterns=alias_patterns,
            detector_client=detector_client,
            detector_model=detector_model,
            detector_prompt=detector_prompt,
            thread_firm_dir=paths["thread_firm_dir"],
            autosave_every=THREAD_AUTOSAVE_EVERY,
        )

    elif RUN_MODE == "rebuild_events_from_existing_threads":
        thread_firm_df = load_existing_thread_firm_pairs(
            thread_firm_dir=paths["thread_firm_dir"],
            explicit_path=EXISTING_THREAD_FIRM_PAIRS_PATH,
        )

    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")

    print(f"Thread x firm pairs ready: {len(thread_firm_df)}")

    event_firm_df = build_event_firm_pairs(
        events_df=events_df,
        thread_firm_df=thread_firm_df,
        alias_patterns=alias_patterns,
        event_firm_dir=paths["event_firm_dir"],
        autosave_every=EVENT_AUTOSAVE_EVERY,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    thread_out = Path(paths["thread_firm_dir"]) / f"thread_firm_pairs_{ts}.parquet"
    event_out = Path(paths["event_firm_dir"]) / f"event_firm_pairs_{ts}.parquet"

    thread_firm_df = validate_thread_firm_df(thread_firm_df)
    event_firm_df = validate_event_firm_df(event_firm_df)

    thread_firm_df.to_parquet(thread_out, index=False)
    event_firm_df.to_parquet(event_out, index=False)

    print("\nSaved:")
    print(thread_out)
    print(event_out)

    print("\nCounts:")
    print("Raw events:", len(events_df))
    print("Thread x firm pairs:", len(thread_firm_df))
    print("Event x firm pairs:", len(event_firm_df))

    print("\nEvent candidate_source distribution:")
    print(event_firm_df["candidate_source"].value_counts(dropna=False))

    print("\nThread mention_type distribution:")
    print(thread_firm_df["mention_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
