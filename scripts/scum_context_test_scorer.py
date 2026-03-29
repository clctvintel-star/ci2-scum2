import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Build scorer-friendly context test file for SCUM2.")
    p.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to event_firm_pairs parquet",
    )
    p.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output parquet",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap after filtering",
    )
    p.add_argument(
        "--min-event-tokens",
        type=int,
        default=4,
        help="Minimum token count for event_text",
    )
    p.add_argument(
        "--require-parent-firm-mention",
        action="store_true",
        help="Require canonical_name or matched_aliases to appear in parent_text",
    )
    return p.parse_args()


def safe_text(x) -> str:
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()


def normalize_text(x: str) -> str:
    x = safe_text(x).lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def token_count(x: str) -> int:
    return len(re.findall(r"\b\w+\b", safe_text(x)))


def is_deleted_or_removed(x: str) -> bool:
    t = normalize_text(x)
    return t in {"[deleted]", "[removed]", "deleted", "removed", ""}


def parse_aliases(raw: str) -> list[str]:
    raw = safe_text(raw)
    if not raw:
        return []
    return [a.strip() for a in raw.split(",") if a.strip()]


def parent_mentions_firm(parent_text: str, canonical_name: str, matched_aliases: str) -> bool:
    pt = normalize_text(parent_text)
    firm = normalize_text(canonical_name)

    if firm and firm in pt:
        return True

    for alias in parse_aliases(matched_aliases):
        if normalize_text(alias) and normalize_text(alias) in pt:
            return True

    return False


def build_augmented_event_text(row: pd.Series) -> str:
    firm = safe_text(row.get("canonical_name"))
    title = safe_text(row.get("thread_title"))
    parent = safe_text(row.get("parent_text"))
    event = safe_text(row.get("event_text"))

    parts = [
        f"TARGET FIRM: {firm}",
    ]

    if title:
        parts.append(f"THREAD TITLE:\n{title}")

    if parent:
        parts.append(f"PARENT CONTEXT:\n{parent}")

    if event:
        parts.append(f"EVENT TEXT:\n{event}")

    return "\n\n".join(parts).strip()


def main():
    args = parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output_file)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_parquet(in_path).copy()
    print("Loaded rows:", len(df))

    required = ["candidate_source", "canonical_name", "event_text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 1) context only
    df = df[df["candidate_source"] == "thread_context"].copy()
    print("After context-only filter:", len(df))

    # 2) parent must exist
    if "parent_text" not in df.columns:
        raise ValueError("Input file missing parent_text column")
    df = df[df["parent_text"].notna()].copy()
    df = df[df["parent_text"].astype(str).str.strip() != ""].copy()
    print("After parent_text exists filter:", len(df))

    # 3) child event text must be nontrivial
    df = df[~df["event_text"].apply(is_deleted_or_removed)].copy()
    df = df[df["event_text"].apply(token_count) >= args.min_event_tokens].copy()
    print("After event_text quality filter:", len(df))

    # 4) optional: parent must mention firm / alias
    if args.require_parent_firm_mention:
        if "matched_aliases" not in df.columns:
            df["matched_aliases"] = ""

        df = df[
            df.apply(
                lambda r: parent_mentions_firm(
                    r.get("parent_text", ""),
                    r.get("canonical_name", ""),
                    r.get("matched_aliases", ""),
                ),
                axis=1,
            )
        ].copy()
        print("After parent-firm-mention filter:", len(df))

    # 5) make scorer-friendly text
    if "thread_title" not in df.columns:
        df["thread_title"] = ""

    df["event_text_original"] = df["event_text"]
    df["event_text"] = df.apply(build_augmented_event_text, axis=1)

    # 6) useful diagnostics
    df["augmented_token_count"] = df["event_text"].apply(token_count)

    # 7) optional cap
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()
        print("After max_rows cap:", len(df))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print("\nSaved:", out_path)
    print("Final rows:", len(df))

    if "canonical_name" in df.columns:
        print("\nTop firms:")
        print(df["canonical_name"].value_counts().head(20))


if __name__ == "__main__":
    main()
