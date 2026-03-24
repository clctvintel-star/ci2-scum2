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
# LOAD
# ============================================

df = pd.read_parquet(INPUT_FILE)

print("Loaded rows:", len(df))

# keep only scored rows
df = df[df["final_scum"].notna()].copy()

print("Scored rows:", len(df))


# ============================================
# BASIC CLEAN
# ============================================

df["date"] = pd.to_datetime(df["created_utc"], unit="s").dt.date

# engagement proxy
df["engagement"] = (
    df.get("score", 0).fillna(0)
    + df.get("num_comments", 0).fillna(0)
)

df["engagement"] = df["engagement"].clip(lower=0)


# ============================================
# POST-LEVEL AGGREGATION
# ============================================

post_group = df.groupby(
    ["canonical_name", "post_id", "date"],
    as_index=False
)

post_days = post_group.agg({
    "final_scum": "mean",
    "event_id": "count",
    "engagement": "sum"
})

post_days.rename(columns={
    "final_scum": "post_scum",
    "event_id": "event_count"
}, inplace=True)

print("Post-days rows:", len(post_days))


# ============================================
# FUND-DAY AGGREGATION
# ============================================

fund_group = post_days.groupby(
    ["canonical_name", "date"],
    as_index=False
)

fund_days = fund_group.agg({
    "post_scum": "mean",
    "post_id": "count",
    "event_count": "sum",
    "engagement": "sum"
})

fund_days.rename(columns={
    "post_scum": "mean_scum",
    "post_id": "post_count"
}, inplace=True)


# ============================================
# ENGAGEMENT-WEIGHTED SCUM
# ============================================

weighted = post_days.copy()
weighted["weighted_scum"] = weighted["post_scum"] * weighted["engagement"]

weighted_group = weighted.groupby(
    ["canonical_name", "date"],
    as_index=False
).agg({
    "weighted_scum": "sum",
    "engagement": "sum"
})

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

fund_days = fund_days.sort_values(["canonical_name", "date"])


# ============================================
# ROLLING AVERAGES
# ============================================

def add_rolling(group):
    group = group.sort_values("date")

    group["ma28"] = group["mean_scum"].rolling(
        window=28,
        min_periods=7
    ).mean()

    group["ma90"] = group["mean_scum"].rolling(
        window=90,
        min_periods=21
    ).mean()

    return group

fund_days = fund_days.groupby(
    "canonical_name",
    group_keys=False
).apply(add_rolling)


# ============================================
# SAVE
# ============================================

Path(POST_DAYS_OUT).parent.mkdir(parents=True, exist_ok=True)
Path(FUND_DAYS_OUT).parent.mkdir(parents=True, exist_ok=True)

post_days.to_parquet(POST_DAYS_OUT, index=False)
fund_days.to_parquet(FUND_DAYS_OUT, index=False)

print("\nSaved:")
print("Post days →", POST_DAYS_OUT)
print("Fund days →", FUND_DAYS_OUT)


# ============================================
# QUICK STATS
# ============================================

print("\nFund summary:")
print(
    fund_days.groupby("canonical_name")["mean_scum"]
    .mean()
    .sort_values(ascending=False)
)
