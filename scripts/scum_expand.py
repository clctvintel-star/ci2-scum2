import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

from utils import load_configs, load_env, ensure_dir, load_prompt


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


def build_thread_text(group):
    texts = []
    group = group.sort_values(["created_utc", "depth"], ascending=[True, True])

    for _, row in group.iterrows():
        txt = row["event_text"]
        if isinstance(txt, str) and txt.strip():
            texts.append(txt)

    return "\n\n".join(texts).strip()


def regex_matches(text, alias_patterns):
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


def safe_json_load(text):
    text = text.strip()

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


def call_thread_detector(client, model_name, prompt, retry_limit=3, sleep_seconds=1):
    for _ in range(retry_limit):
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
            print(f"LLM thread detector error, retrying: {e}")
            time.sleep(sleep_seconds)

    return []


def build_detector_prompt(prompt_template, canonical_names, thread_title, thread_text, regex_candidates):
    return prompt_template.format(
        FIRM_UNIVERSE=json.dumps(canonical_names, ensure_ascii=False),
        REGEX_CANDIDATES=json.dumps(regex_candidates, ensure_ascii=False),
        THREAD_TITLE=thread_title or "",
        THREAD_TEXT=thread_text or "",
    )


def normalize_llm_candidates(llm_output, allowed_names):
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

        mention_type = item.get("mention_type")
        confidence = item.get("confidence")
        rationale = item.get("rationale")

        try:
            confidence = float(confidence)
        except Exception:
            confidence = None

        rows.append({
            "canonical_name": canonical_name,
            "mention_type": mention_type,
            "llm_confidence": confidence,
            "llm_rationale": rationale,
        })

    return rows


def build_thread_firm_pairs(events_df, alias_patterns, detector_client, detector_model, detector_prompt):
    rows = []
    grouped = events_df.groupby("post_id", dropna=True)
    canonical_names = [f["canonical_name"] for f in alias_patterns]

    for post_id, group in grouped:
        post_rows = group[group["type"] == "post"].copy()
        if post_rows.empty:
            continue

        post_row = post_rows.iloc[0]
        thread_title = post_row.get("title")
        thread_created_utc = post_row.get("created_utc")
        subreddit = post_row.get("subreddit")
        thread_text = build_thread_text(group)

        regex_hits = regex_matches(thread_text, alias_patterns)
        regex_candidate_names = [x["canonical_name"] for x in regex_hits]
        regex_alias_map = {x["canonical_name"]: x["matched_aliases"] for x in regex_hits}

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
        final_names = sorted(set(regex_candidate_names) | set(llm_map.keys()))

        for canonical_name in final_names:
            regex_aliases = regex_alias_map.get(canonical_name, [])
            llm_info = llm_map.get(canonical_name, {})

            rows.append({
                "post_id": post_id,
                "canonical_name": canonical_name,
                "matched_aliases": ", ".join(regex_aliases) if regex_aliases else None,
                "thread_direct_match": canonical_name in regex_alias_map,
                "thread_text": thread_text,
                "thread_title": thread_title,
                "subreddit": subreddit,
                "thread_created_utc": thread_created_utc,
                "candidate_strength": llm_info.get("llm_confidence"),
                "mention_type": llm_info.get("mention_type"),
                "llm_rationale": llm_info.get("llm_rationale"),
            })

        print(f"thread {post_id}: regex={len(regex_hits)} llm={len(llm_hits)} final={len(final_names)}")

    if not rows:
        return pd.DataFrame(columns=[
            "post_id", "canonical_name", "matched_aliases", "thread_direct_match",
            "thread_text", "thread_title", "subreddit", "thread_created_utc",
            "candidate_strength", "mention_type", "llm_rationale"
        ])

    return pd.DataFrame(rows).drop_duplicates(subset=["post_id", "canonical_name"])


def build_event_firm_pairs(events_df, thread_firm_df, alias_patterns):
    rows = []

    thread_firm_map = {}
    for _, row in thread_firm_df.iterrows():
        thread_firm_map.setdefault(row["post_id"], set()).add(row["canonical_name"])

    for _, row in events_df.iterrows():
        event_text = row["event_text"]
        direct_matches = regex_matches(event_text, alias_patterns)
        direct_firms = {m["canonical_name"]: m["matched_aliases"] for m in direct_matches}

        thread_firms = thread_firm_map.get(row["post_id"], set())

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
    env = load_env(paths["env_path"])

    ensure_dir(paths["thread_firm_dir"])
    ensure_dir(paths["event_firm_dir"])

    detector_model = settings["models"]["expand"]["thread_detector"]
    detector_prompt = load_prompt(str(Path(__file__).resolve().parents[1] / "prompts" / "relevance_prompt.txt"))

    if not env["ANTHROPIC_API_KEY"]:
        raise ValueError("ANTHROPIC_API_KEY missing from env")

    detector_client = Anthropic(api_key=env["ANTHROPIC_API_KEY"])

    alias_patterns = build_alias_patterns(firms)
    events_df = load_latest_events(paths["events_dir"]).copy()
    events_df["event_text"] = events_df.apply(assemble_event_text, axis=1)

    thread_firm_df = build_thread_firm_pairs(
        events_df=events_df,
        alias_patterns=alias_patterns,
        detector_client=detector_client,
        detector_model=detector_model,
        detector_prompt=detector_prompt,
    )

    event_firm_df = build_event_firm_pairs(
        events_df=events_df,
        thread_firm_df=thread_firm_df,
        alias_patterns=alias_patterns,
    )

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
