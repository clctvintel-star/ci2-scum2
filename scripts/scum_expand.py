import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils import load_configs, ensure_dir


def build_alias_patterns(firms_config):
    patterns = []

    for firm in firms_config["firms"]:
        canonical_name = firm["canonical_name"]
        aliases = firm.get("aliases", [])
        risky_aliases = set(a.lower() for a in firm.get("risky_aliases", []))

        compiled = []
        for alias in aliases:
            alias_clean = alias.strip()
            if not alias_clean:
                continue

            if alias_clean.lower() in risky_aliases or len(alias_clean) <= 3:
                pattern = re.compile(rf"(?<!\w){re.escape(alias_clean)}(?!\w)", flags=re.IGNORECASE)
            else:
                pattern = re.compile(rf"\b{re.escape(alias_clean)}\b", flags=re.IGNORECASE)

            compiled.append((alias_clean, pattern))

        patterns.append({
            "canonical_name": canonical_name,
            "patterns": compiled,
        })

    return patterns


def load_latest_events(events_dir):
    files = sorted(Path(events_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {events_dir}")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["event_id"]).copy()
    return df


def assemble_event_text(row):
    parts = []
    for col in ["title", "selftext", "body"]:
        val = row.get(col)
        if pd.notna(val) and val:
            parts.append(str(val))
    return "\n\n".join(parts).strip()


def find_matches(text, alias_patterns):
    matched = []

    if not isinstance(text, str) or not text.strip():
        return matched

    for firm in alias_patterns:
        firm_matches = []

        for alias, pattern in firm["patterns"]:
            if pattern.search(text):
                firm_matches.append(alias)

        if firm_matches:
            matched.append({
                "canonical_name": firm["canonical_name"],
                "matched_aliases": sorted(set(firm_matches)),
            })

    return matched


def build_thread_text(group):
    texts = []
    group = group.sort_values(["created_utc", "depth"], ascending=[True, True])

    for _, row in group.iterrows():
        txt = row["event_text"]
        if isinstance(txt, str) and txt.strip():
            texts.append(txt)

    return "\n\n".join(texts).strip()


def build_thread_firm_pairs(events_df, alias_patterns):
    rows = []

    grouped = events_df.groupby("post_id", dropna=True)

    for post_id, group in grouped:
        post_rows = group[group["type"] == "post"].copy()
        if post_rows.empty:
            continue

        post_row = post_rows.iloc[0]
        thread_title = post_row.get("title")
        thread_created_utc = post_row.get("created_utc")
        subreddit = post_row.get("subreddit")

        thread_text = build_thread_text(group)
        matches = find_matches(thread_text, alias_patterns)

        for m in matches:
            rows.append({
                "post_id": post_id,
                "canonical_name": m["canonical_name"],
                "matched_aliases": ", ".join(m["matched_aliases"]),
                "thread_direct_match": True,
                "thread_text": thread_text,
                "thread_title": thread_title,
                "subreddit": subreddit,
                "thread_created_utc": thread_created_utc,
                "candidate_strength": 1.0,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "post_id", "canonical_name", "matched_aliases", "thread_direct_match",
            "thread_text", "thread_title", "subreddit", "thread_created_utc", "candidate_strength"
        ])

    return pd.DataFrame(rows).drop_duplicates(subset=["post_id", "canonical_name"])


def build_event_firm_pairs(events_df, thread_firm_df, alias_patterns):
    rows = []

    thread_firm_map = {}
    for _, row in thread_firm_df.iterrows():
        thread_firm_map.setdefault(row["post_id"], set()).add(row["canonical_name"])

    for _, row in events_df.iterrows():
        event_text = row["event_text"]
        direct_matches = find_matches(event_text, alias_patterns)
        direct_firms = {m["canonical_name"]: m["matched_aliases"] for m in direct_matches}

        thread_firms = thread_firm_map.get(row["post_id"], set())

        # direct event matches
        for canonical_name, aliases in direct_firms.items():
            rows.append({
                "event_id": row["event_id"],
                "post_id": row["post_id"],
                "comment_id": row["comment_id"],
                "parent_id": row["parent_id"],
                "canonical_name": canonical_name,
                "matched_aliases": ", ".join(sorted(set(aliases))),
                "candidate_source": "direct_event_match",
                "event_text": event_text,
                "created_utc": row["created_utc"],
                "depth": row["depth"],
                "score": row["score"],
                "subreddit": row["subreddit"],
            })

        # thread-level inherited candidates
        for canonical_name in thread_firms:
            if canonical_name in direct_firms:
                continue

            rows.append({
                "event_id": row["event_id"],
                "post_id": row["post_id"],
                "comment_id": row["comment_id"],
                "parent_id": row["parent_id"],
                "canonical_name": canonical_name,
                "matched_aliases": None,
                "candidate_source": "thread_context",
                "event_text": event_text,
                "created_utc": row["created_utc"],
                "depth": row["depth"],
                "score": row["score"],
                "subreddit": row["subreddit"],
            })

    if not rows:
        return pd.DataFrame(columns=[
            "event_id", "post_id", "comment_id", "parent_id", "canonical_name",
            "matched_aliases", "candidate_source", "event_text", "created_utc",
            "depth", "score", "subreddit"
        ])

    return pd.DataFrame(rows).drop_duplicates(subset=["event_id", "canonical_name"])


def main():
    firms, paths, settings = load_configs()

    ensure_dir(paths["thread_firm_dir"])
    ensure_dir(paths["event_firm_dir"])

    alias_patterns = build_alias_patterns(firms)
    events_df = load_latest_events(paths["events_dir"]).copy()

    events_df["event_text"] = events_df.apply(assemble_event_text, axis=1)

    thread_firm_df = build_thread_firm_pairs(events_df, alias_patterns)
    event_firm_df = build_event_firm_pairs(events_df, thread_firm_df, alias_patterns)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    thread_out = Path(paths["thread_firm_dir"]) / f"thread_firm_pairs_{ts}.parquet"
    event_out = Path(paths["event_firm_dir"]) / f"event_firm_pairs_{ts}.parquet"

    thread_firm_df.to_parquet(thread_out, index=False)
    event_firm_df.to_parquet(event_out, index=False)

    print("\nSaved:")
    print(thread_out)
    print(event_out)
    print("\nCounts:")
    print("Raw events:", len(events_df))
    print("Thread x firm pairs:", len(thread_firm_df))
    print("Event x firm pairs:", len(event_firm_df))


if __name__ == "__main__":
    main()
