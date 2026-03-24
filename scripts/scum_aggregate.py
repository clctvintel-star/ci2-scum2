import pandas as pd
import numpy as np
from pathlib import Path

# ============================================
# CONFIG
# ============================================

INPUT_FILE = "/content/drive/MyDrive/CI2/db/scum2/scored_events/scored_events_direct_resume_v1_latest.parquet"

POST_DAYS_OUT = "/content/drive/MyDrive/CI2/db/scum2/post_days/post_days_latest.parquet"
FUND_DAYS_OUT = "/content/drive/MyDrive/CI2/db/scum2/fund_days/fund_days_latest.parquet"


# ============================================
# HELPERS
# ============================================

def safe_series(df, col, default=0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def add_rolling(group):
    group = group.sort_values("date").copy()

    group["ma28"] = group["mean_scum"].rolling(
        window=28,
        min_periods=7
    ).mean()

    group["ma90"] = group["mean_scum"].rolling(
        window=90,
        min_periods=21
    ).mean()

    return group


# ============================================
# LOAD
# ============================================

df = pd.read_parquet(INPUT_FILE).copy()

print("Loaded rows:", len(df))

# keep only scored rows
df = df[df["final_scum"].notna()].copy()

print("Scored rows:", len(df))


# ============================================
# BASIC CLEAN
# ============================================

# required fields check
required_cols = ["canonical_name", "post_id", "created_utc", "final_scum"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["final_scum"] = pd.to_numeric(df["final_scum"], errors="coerce")
df = df[df["final_scum"].notna()].copy()

df["date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce").dt.date
df = df[df["date"].notna()].copy()

# robust engagement proxy
score = safe_series(df, "score", default=0)
num_comments = safe_series(df, "num_comments", default=0)

df["engagement"] = (score + num_comments).clip(lower=0)

# fallback event_id if missing
if "event_id" not in df.columns:
    df["event_id"] = df.index.astype(str)


# ============================================
# POST-LEVEL AGGREGATION
# ============================================

post_days = (
    df.groupby(["canonical_name", "post_id", "date"], as_index=False)
    .agg(
        post_scum=("final_scum", "mean"),
        event_count=("event_id", "count"),
        engagement=("engagement", "sum"),
    )
)

print("Post-days rows:", len(post_days))


# ============================================
# FUND-DAY AGGREGATION
# ============================================

fund_days = (
    post_days.groupby(["canonical_name", "date"], as_index=False)
    .agg(
        mean_scum=("post_scum", "mean"),
        median_scum=("post_scum", "median"),
        post_count=("post_id", "count"),
        event_count=("event_count", "sum"),
        engagement=("engagement", "sum"),
    )
)


# ============================================
# ENGAGEMENT-WEIGHTED SCUM
# ============================================

weighted = post_days.copy()
weighted["weighted_scum"] = weighted["post_scum"] * weighted["engagement"]

weighted_group = (
    weighted.groupby(["canonical_name", "date"], as_index=False)
    .agg(
        weighted_scum=("weighted_scum", "sum"),
        engagement=("engagement", "sum"),
    )
)

weighted_group["engagement_weighted_scum"] = (
    weighted_group["weighted_scum"] /
    weighted_group["engagement"].replace(0, np.nan)
)

fund_days = fund_days.merge(
    weighted_group[["canonical_name", "date", "engagement_weighted_scum"]],
    on=["canonical_name", "date"],
    how="left"
)


# ============================================
# SORT
# ============================================

fund_days = fund_days.sort_values(["canonical_name", "date"]).reset_index(drop=True)


# ============================================
# ROLLING AVERAGES
# ============================================

fund_days = (
    fund_days.groupby("canonical_name", group_keys=False)
    .apply(add_rolling)
    .reset_index(drop=True)
)


# ============================================
# SAVE
# ============================================

Path(POST_DAYS_OUT).parent.mkdir(parents=True, exist_ok=True)
Path(FUND_DAYS_OUT).parent.mkdir(parents=True, exist_ok=True)

post_days.to_parquet(POST_DAYS_OUT, index=False)
fund_days.to_parquet(FUND_DAYS_OUT, index=False)

print("\nSaved:")
print("Post days ->", POST_DAYS_OUT)
print("Fund days ->", FUND_DAYS_OUT)


# ============================================
# QUICK STATS
# ============================================

print("\nFund summary:")
print(
    fund_days.groupby("canonical_name")["mean_scum"]
    .mean()
    .sort_values(ascending=False)
)

print("\nDate range:")
print(fund_days["date"].min(), "to", fund_days["date"].max())

print("\nRows:")
print("post_days:", len(post_days))
print("fund_days:", len(fund_days))
