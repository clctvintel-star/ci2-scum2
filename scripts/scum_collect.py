import praw
import pandas as pd
import time
from serpapi import GoogleSearch
from pathlib import Path
from datetime import datetime, timezone

from utils import load_configs, load_env, ensure_dir


# --------------------------------------------------
# helpers
# --------------------------------------------------

def build_reddit_client(env):

    reddit = praw.Reddit(
        client_id=env["REDDIT_CLIENT_ID"],
        client_secret=env["REDDIT_SECRET"],
        user_agent=env["REDDIT_AGENT"],
    )

    return reddit


def search_reddit_urls(query, serpapi_key, max_pages, num):

    urls = []

    for page in range(max_pages):

        params = {
            "engine": "google",
            "q": f"{query} site:reddit.com",
            "api_key": serpapi_key,
            "num": num,
            "start": page * num,
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if "organic_results" not in results:
            break

        for r in results["organic_results"]:
            link = r.get("link")
            if "reddit.com" in link:
                urls.append(link)

        time.sleep(1)

    return list(set(urls))


def fetch_thread(reddit, url):

    submission = reddit.submission(url=url)
    submission.comments.replace_more(limit=0)

    rows = []

    # post
    rows.append({
        "event_id": f"post_{submission.id}",
        "post_id": submission.id,
        "comment_id": None,
        "parent_id": None,
        "type": "post",
        "subreddit": submission.subreddit.display_name,
        "created_utc": submission.created_utc,
        "depth": 0,
        "author": str(submission.author),
        "score": submission.score,
        "title": submission.title,
        "selftext": submission.selftext,
        "body": None,
        "url": submission.url,
        "permalink": submission.permalink
    })

    # comments
    for comment in submission.comments.list():

        depth = getattr(comment, "depth", None)

        rows.append({
            "event_id": f"comment_{comment.id}",
            "post_id": submission.id,
            "comment_id": comment.id,
            "parent_id": comment.parent_id,
            "type": "comment",
            "subreddit": submission.subreddit.display_name,
            "created_utc": comment.created_utc,
            "depth": depth,
            "author": str(comment.author),
            "score": comment.score,
            "title": None,
            "selftext": None,
            "body": comment.body,
            "url": submission.url,
            "permalink": comment.permalink
        })

    return rows


# --------------------------------------------------
# main
# --------------------------------------------------

def main():

    firms, paths, settings = load_configs()

    env = load_env(paths["env_path"])

    reddit = build_reddit_client(env)

    ensure_dir(paths["events_dir"])

    serpapi_key = env["SERPAPI_API_KEY"]

    all_rows = []

    for firm in firms["firms"]:

        name = firm["canonical_name"]

        print(f"\nSearching for: {name}")

        urls = search_reddit_urls(
            query=name,
            serpapi_key=serpapi_key,
            max_pages=settings["collection"]["max_serpapi_pages"],
            num=settings["collection"]["serpapi_num"],
        )

        print("threads found:", len(urls))

        for url in urls:

            try:

                rows = fetch_thread(reddit, url)
                all_rows.extend(rows)

                print("events:", len(rows))

            except Exception as e:

                print("error:", e)

    df = pd.DataFrame(all_rows)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    outfile = Path(paths["events_dir"]) / f"scum_events_{ts}.parquet"

    df.to_parquet(outfile, index=False)

    print("\nSaved:", outfile)
    print("Rows:", len(df))


if __name__ == "__main__":
    main()
