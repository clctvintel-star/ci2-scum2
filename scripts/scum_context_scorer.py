import pandas as pd
import numpy as np
from pathlib import Path

# ============================================
# CONFIG
# ============================================

SCORED_DIRECT = "/content/drive/MyDrive/CI2/db/scum2/scored_events/scored_events_direct_resume_v1_latest.parquet"
SCORED_CONTEXT = "/content/drive/MyDrive/CI2/db/scum2/scored_events/scored_events_context_TEST.parquet"

OUTPUT_DIR = "/content/drive/MyDrive/CI2/db/scum2/context_experiment/"

FIRMS = ["Two Sigma", "Jane Street"]
SAMPLE_PER_FIRM = 2500

POS_THRESHOLD = 0.10
NEG_THRESHOLD = -0.10

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================
# LOAD
# ============================================

direct_df = pd.read_parquet(SCORED_DIRECT)
context_df = pd.read_parquet(SCORED_CONTEXT)

print("Direct rows:", len(direct_df))
print("Context rows:", len(context_df))

# keep only valid scored rows
direct_df = direct_df[direct_df["final_scum"].notna()].copy()
context_df = context_df[context_df["final_scum"].notna()].copy()

# ============================================
# SAMPLE CONTEXT
# ============================================

context_sample = (
    context_df[context_df["canonical_name"].isin(FIRMS)]
    .groupby("canonical_name", group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), SAMPLE_PER_FIRM), random_state=42))
    .reset_index(drop=True)
)

print("Sampled context rows:", len(context_sample))

# ============================================
# BUILD TWO DATASETS
# ============================================

direct_only = direct_df.copy()

direct_plus_context = pd.concat([
    direct_df,
    context_sample
], ignore_index=True)

# ============================================
# AGG FUNCTION (REUSE YOUR LOGIC)
# ============================================

def build_fund_days(df, label):

    df = df.copy()

    df["date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce").dt.date
    df = df[df["date"].notna()].copy()

    df["pos_signal"] = (df["final_scum"] >= POS_THRESHOLD).astype(int)
    df["neg_signal"] = (df["final_scum"] <= NEG_THRESHOLD).astype(int)
    df["net_signal"] = df["pos_signal"] - df["neg_signal"]

    # POST LEVEL
    post_days = (
        df.groupby(["canonical_name", "post_id", "date"], as_index=False)
        .agg(
            post_scum=("final_scum", "mean"),
            event_count=("event_id", "count"),
            pos_event_count=("pos_signal", "sum"),
            neg_event_count=("neg_signal", "sum"),
            net_signal_sum=("net_signal", "sum"),
        )
    )

    post_days["dsr_pos"] = post_days["pos_event_count"] / post_days["event_count"]
    post_days["dsr_neg"] = post_days["neg_event_count"] / post_days["event_count"]
    post_days["dsr_net"] = post_days["dsr_pos"] - post_days["dsr_neg"]

    # FUND LEVEL
    fund_days = (
        post_days.groupby(["canonical_name", "date"], as_index=False)
        .agg(
            mean_scum=("post_scum", "mean"),
            post_count=("post_id", "count"),
            event_count=("event_count", "sum"),
            pos_event_count=("pos_event_count", "sum"),
            neg_event_count=("neg_event_count", "sum"),
        )
    )

    fund_days["dsr_pos"] = fund_days["pos_event_count"] / fund_days["event_count"]
    fund_days["dsr_neg"] = fund_days["neg_event_count"] / fund_days["event_count"]
    fund_days["dsr_net"] = fund_days["dsr_pos"] - fund_days["dsr_neg"]

    # rolling
    fund_days = fund_days.sort_values(["canonical_name", "date"])

    fund_days["ma28"] = (
        fund_days.groupby("canonical_name")["mean_scum"]
        .transform(lambda x: x.rolling(28, min_periods=7).mean())
    )

    out_path = f"{OUTPUT_DIR}/fund_days_{label}.parquet"
    fund_days.to_parquet(out_path, index=False)

    print(f"\nSaved: {out_path}")
    return fund_days

# ============================================
# RUN BOTH
# ============================================

fd_direct = build_fund_days(direct_only, "direct_only")
fd_plus = build_fund_days(direct_plus_context, "direct_plus_context")

# ============================================
# QUICK DIFF CHECK
# ============================================

for f in FIRMS:
    d1 = fd_direct[fd_direct["canonical_name"] == f]["mean_scum"].mean()
    d2 = fd_plus[fd_plus["canonical_name"] == f]["mean_scum"].mean()

    print(f"\n{f}")
    print("Direct mean:", round(d1, 4))
    print("With context:", round(d2, 4))
