import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pandas as pd
import praw
from serpapi import GoogleSearch

from utils import load_configs, load_env, ensure_dir


LEDGER_COLUMNS = [
    "url",
    "discovered_for",
    "discovered_alias",
    "discovered_at_utc",
    "last_seen_utc",
    "post_id",
    "subreddit",
    "status",
    "collected_at_utc",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(description="SCUM2 Reddit collector")
    parser.add_argument("--fund", type=str, default=None, help="Collect for one canonical fund only")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--reset-ledger",
        action="store_true",
        help="Delete existing discovery ledger before run",
    )
    return parser.parse_args()


def to_timestamp(date_str, end_of_day=False):
    if not date_str:
        return None

    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        return dt.timestamp() + 86399
    return dt.timestamp()


def normalize_reddit_url(url):
    if not url:
        return None

    parsed = urlparse(url)
    netloc = (parsed.netloc or "").lower()

    if "reddit.com" not in netloc:
        return None

    path = (parsed.path or "").rstrip("/")
    if "/comments/" not in path:
        return None

    clean = parsed._replace(
        scheme="https",
        netloc="www.reddit.com",
        params="",
        query="",
        fragment="",
    )
    return urlunparse(clean).rstrip("/")


def extract_subreddit_from_url(url):
    if not url:
        return None

    try:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0].lower() == "r":
            return parts[1]
    except Exception:
        return None

    return None


def build_reddit_client(env):
    return praw.Reddit(
        client_id=env["REDDIT_CLIENT_ID"],
        client_secret=env["REDDIT_SECRET"],
        user_agent=env["REDDIT_AGENT"],
    )


def read_ledger(ledger_path):
    ledger_path = Path(ledger_path)
    if ledger_path.exists():
        df = pd.read_parquet(ledger_path)
        for col in LEDGER_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[LEDGER_COLUMNS].copy()
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def write_ledger(df, ledger_path):
    ledger_path = Path(ledger_path)
    ensure_dir(ledger_path.parent)
    df.to_parquet(ledger_path, index=False)


def maybe_reset_ledger(ledger_path, reset=False):
    ledger_path = Path(ledger_path)
    if reset and ledger_path.exists():
        ledger_path.unlink()
        print(f"RESET ledger: {ledger_path}")


def upsert_discoveries(ledger_df, discovered_rows):
    if not discovered_rows:
        return ledger_df

    now_utc = datetime.now(timezone.utc).isoformat()
    discovered_df = pd.DataFrame(discovered_rows).drop_duplicates(subset=["url"]).copy()

    if ledger_df.empty:
        discovered_df["last_seen_utc"] = discovered_df["discovered_at_utc"].fillna(now_utc)
        discovered_df["post_id"] = None
        discovered_df["subreddit"] = None
        discovered_df["status"] = "new"
        discovered_df["collected_at_utc"] = None
        discovered_df["error"] = None
        return discovered_df[LEDGER_COLUMNS]

    existing_urls = set(ledger_df["url"].dropna().tolist())
    new_rows = discovered_df[~discovered_df["url"].isin(existing_urls)].copy()

    if not new_rows.empty:
        new_rows["last_seen_utc"] = new_rows["discovered_at_utc"].fillna(now_utc)
        new_rows["post_id"] = None
        new_rows["subreddit"] = None
        new_rows["status"] = "new"
        new_rows["collected_at_utc"] = None
        new_rows["error"] = None
        ledger_df = pd.concat([ledger_df, new_rows[LEDGER_COLUMNS]], ignore_index=True)

    seen_map = discovered_df.groupby("url")["discovered_at_utc"].max().to_dict()
    ledger_df["last_seen_utc"] = ledger_df["url"].map(seen_map).fillna(ledger_df["last_seen_utc"])

    return ledger_df


def search_reddit_urls_for_subreddit(alias, subreddit, serpapi_key, max_pages):
    urls = []
    seen_urls = set()

    query = f"\"{alias}\" site:reddit.com/r/{subreddit}"

    for page in range(max_pages):
        start = page * 10

        params = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_key,
            "num": 10,
            "start": start,
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        organic = results.get("organic_results", [])

        if not organic:
            print(f"    {subreddit} | page {page + 1}/{max_pages} | organic=0 | stopping")
            break

        page_candidate_count = 0
        page_new_unique = 0

        for r in organic:
            link = normalize_reddit_url(r.get("link"))
            if not link:
                continue

            link_subreddit = extract_subreddit_from_url(link)
            if not link_subreddit or link_subreddit.lower() != subreddit.lower():
                continue

            page_candidate_count += 1

            if link not in seen_urls:
                seen_urls.add(link)
                urls.append(link)
                page_new_unique += 1

        print(
            f"    {subreddit} | page {page + 1}/{max_pages} | "
            f"organic={len(organic)} | comment_threads={page_candidate_count} | "
            f"new_urls={page_new_unique} | running_total={len(urls)}"
        )

        time.sleep(1)

    return urls


def fetch_thread(
    reddit,
    url,
    discovered_for,
    discovered_alias,
    depth_cap=3,
    window_days=60,
    score_min=1,
    allowed_subreddits=None,
    start_ts=None,
    end_ts=None,
):
    submission = reddit.submission(url=url)
    submission.comments.replace_more(limit=0)

    subreddit_name = submission.subreddit.display_name
    subreddit_lower = subreddit_name.lower()

    if allowed_subreddits and subreddit_lower not in allowed_subreddits:
        return [], {
            "status": "skipped_subreddit",
            "post_id": submission.id,
            "subreddit": subreddit_name,
            "error": None,
        }

    post_created = submission.created_utc

    if start_ts is not None and post_created < start_ts:
        return [], {
            "status": "skipped_date",
            "post_id": submission.id,
            "subreddit": subreddit_name,
            "error": None,
        }

    if end_ts is not None and post_created > end_ts:
        return [], {
            "status": "skipped_date",
            "post_id": submission.id,
            "subreddit": subreddit_name,
            "error": None,
        }

    cutoff_utc = post_created + (window_days * 86400)
    rows = []

    rows.append({
        "event_id": f"post_{submission.id}",
        "post_id": submission.id,
        "comment_id": None,
        "parent_id": None,
        "type": "post",
        "subreddit": subreddit_name,
        "created_utc": submission.created_utc,
        "depth": 0,
        "author": str(submission.author) if submission.author is not None else None,
        "score": submission.score,
        "title": submission.title,
        "selftext": submission.selftext,
        "body": None,
        "url": submission.url,
        "permalink": f"https://www.reddit.com{submission.permalink}",
        "discovered_for": discovered_for,
        "discovered_alias": discovered_alias,
    })

    for comment in submission.comments.list():
        depth = getattr(comment, "depth", None)
        if depth is None:
            continue
        if depth > depth_cap:
            continue
        if comment.created_utc > cutoff_utc:
            continue

        comment_score = comment.score if comment.score is not None else 0
        if comment_score < score_min:
            continue

        rows.append({
            "event_id": f"comment_{comment.id}",
            "post_id": submission.id,
            "comment_id": comment.id,
            "parent_id": comment.parent_id,
            "type": "comment",
            "subreddit": subreddit_name,
            "created_utc": comment.created_utc,
            "depth": depth,
            "author": str(comment.author) if comment.author is not None else None,
            "score": comment_score,
            "title": None,
            "selftext": None,
            "body": comment.body,
            "url": submission.url,
            "permalink": f"https://www.reddit.com{comment.permalink}",
            "discovered_for": discovered_for,
            "discovered_alias": discovered_alias,
        })

    return rows, {
        "status": "collected",
        "post_id": submission.id,
        "subreddit": subreddit_name,
        "error": None,
    }


def build_run_output_path(events_dir, fund_label):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_label = fund_label.lower().replace(" ", "_").replace(".", "")
    return Path(events_dir) / f"scum_events_{safe_label}_{ts}.parquet"


def save_rows_snapshot(rows_buffer, outfile):
    if not rows_buffer:
        return 0

    df = pd.DataFrame(rows_buffer).drop_duplicates(subset=["event_id"]).copy()
    ensure_dir(Path(outfile).parent)
    df.to_parquet(outfile, index=False)
    return len(df)


def main():
    args = parse_args()

    firms, paths, settings = load_configs()
    env = load_env(paths["env_path"])

    reddit = build_reddit_client(env)
    serpapi_key = env["SERPAPI_API_KEY"]

    if not serpapi_key:
        raise ValueError("SERPAPI_API_KEY missing from env")

    ensure_dir(paths["events_dir"])
    ensure_dir(Path(paths["ledger_path"]).parent)

    maybe_reset_ledger(paths["ledger_path"], reset=args.reset_ledger)
    ledger = read_ledger(paths["ledger_path"])

    allowed_subreddits = [s for s in settings.get("subreddits", [])]
    allowed_subreddits_lower = {s.lower() for s in allowed_subreddits}

    depth_cap = settings["collection"]["depth_cap"]
    window_days = settings["collection"]["window_days"]
    score_min = settings["collection"]["score_min"]
    max_pages = settings["collection"]["max_serpapi_pages"]
    autosave_every_n_posts = settings["collection"]["autosave_every_n_posts"]

    start_ts = to_timestamp(args.start_date, end_of_day=False)
    end_ts = to_timestamp(args.end_date, end_of_day=True)

    if args.fund:
        selected_firms = [f for f in firms["firms"] if f["canonical_name"].lower() == args.fund.lower()]
        if not selected_firms:
            raise ValueError(f"Fund not found in firms.yaml: {args.fund}")
    else:
        selected_firms = firms["firms"]

    fund_label = args.fund if args.fund else "all_firms"
    run_outfile = build_run_output_path(paths["events_dir"], fund_label)

    discovered_rows = []

    for firm in selected_firms:
        canonical_name = firm["canonical_name"]
        aliases = firm.get("aliases", [])

        print(f"\n=== DISCOVERY: {canonical_name} ===")

        for alias in aliases:
            alias_urls = []

            try:
                for subreddit in allowed_subreddits:
                    urls = search_reddit_urls_for_subreddit(
                        alias=alias,
                        subreddit=subreddit,
                        serpapi_key=serpapi_key,
                        max_pages=max_pages,
                    )
                    alias_urls.extend(urls)

                alias_urls = sorted(set(alias_urls))
                print(f"alias='{alias}' -> {len(alias_urls)} usable urls")

                now_utc = datetime.now(timezone.utc).isoformat()
                for url in alias_urls:
                    discovered_rows.append({
                        "url": url,
                        "discovered_for": canonical_name,
                        "discovered_alias": alias,
                        "discovered_at_utc": now_utc,
                    })

            except Exception as e:
                print(f"discovery_error | fund={canonical_name} | alias={alias} | error={e}")

    ledger = upsert_discoveries(ledger, discovered_rows)
    write_ledger(ledger, paths["ledger_path"])

    pending_statuses = {"new", "error"}
    selected_names = {f["canonical_name"] for f in selected_firms}
    pending = ledger[
        ledger["status"].fillna("new").isin(pending_statuses)
        & ledger["discovered_for"].isin(selected_names)
    ].copy()

    print(f"\nPending urls to collect: {len(pending)}")

    rows_buffer = []
    posts_since_flush = 0

    status_counts = {
        "collected": 0,
        "skipped_subreddit": 0,
        "skipped_date": 0,
        "error": 0,
    }

    processed = 0
    report_every = 25

    for idx, row in pending.iterrows():
        url = row["url"]

        try:
            rows, meta = fetch_thread(
                reddit=reddit,
                url=url,
                discovered_for=row["discovered_for"],
                discovered_alias=row["discovered_alias"],
                depth_cap=depth_cap,
                window_days=window_days,
                score_min=score_min,
                allowed_subreddits=allowed_subreddits_lower,
                start_ts=start_ts,
                end_ts=end_ts,
            )

            ledger.at[idx, "post_id"] = meta["post_id"]
            ledger.at[idx, "subreddit"] = meta["subreddit"]
            ledger.at[idx, "status"] = meta["status"]
            ledger.at[idx, "collected_at_utc"] = datetime.now(timezone.utc).isoformat()
            ledger.at[idx, "error"] = meta["error"]

            status_counts[meta["status"]] = status_counts.get(meta["status"], 0) + 1

            if rows:
                rows_buffer.extend(rows)
                print(f"collected: {url} -> {len(rows)} events")
            else:
                print(f"{meta['status']}: {url} -> 0 events")

            posts_since_flush += 1
            processed += 1

            if processed % report_every == 0:
                print(
                    "PROGRESS | "
                    f"processed={processed}/{len(pending)} | "
                    f"collected={status_counts.get('collected', 0)} | "
                    f"skipped_subreddit={status_counts.get('skipped_subreddit', 0)} | "
                    f"skipped_date={status_counts.get('skipped_date', 0)} | "
                    f"errors={status_counts.get('error', 0)} | "
                    f"buffered_rows={len(rows_buffer)}"
                )

            if posts_since_flush >= autosave_every_n_posts:
                nrows = save_rows_snapshot(rows_buffer, run_outfile)
                print(f"AUTOSAVED: {run_outfile} ({nrows} rows)")
                posts_since_flush = 0
                write_ledger(ledger, paths["ledger_path"])

        except Exception as e:
            ledger.at[idx, "status"] = "error"
            ledger.at[idx, "error"] = str(e)
            ledger.at[idx, "collected_at_utc"] = datetime.now(timezone.utc).isoformat()
            status_counts["error"] += 1
            processed += 1
            print(f"error: {url} -> {e}")

            if processed % report_every == 0:
                print(
                    "PROGRESS | "
                    f"processed={processed}/{len(pending)} | "
                    f"collected={status_counts.get('collected', 0)} | "
                    f"skipped_subreddit={status_counts.get('skipped_subreddit', 0)} | "
                    f"skipped_date={status_counts.get('skipped_date', 0)} | "
                    f"errors={status_counts.get('error', 0)} | "
                    f"buffered_rows={len(rows_buffer)}"
                )

    write_ledger(ledger, paths["ledger_path"])

    final_rows = save_rows_snapshot(rows_buffer, run_outfile)
    if final_rows > 0:
        print(f"\nFINAL SAVE: {run_outfile} ({final_rows} rows)")
    else:
        print("\nFINAL SAVE: no rows written")

    print("\nDone.")
    print("Output file:", run_outfile)
    print("Total rows written:", final_rows)
    print("Collected:", status_counts.get("collected", 0))
    print("Skipped subreddit:", status_counts.get("skipped_subreddit", 0))
    print("Skipped date:", status_counts.get("skipped_date", 0))
    print("Errors:", status_counts.get("error", 0))


if __name__ == "__main__":
    main()
