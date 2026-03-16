# SCUM2 Brief

SCUM2 is a time-aware Reddit reputation pipeline for hedge funds.

## Goal

Capture Reddit posts, comments, and replies as timestamped events, score their sentiment and confidence, and aggregate them into daily fund-level sentiment series.

## Key difference from SCUM1

SCUM1 largely treated a thread as a collapsed package and effectively anchored sentiment to the original post date.

SCUM2 fixes that by treating each post, comment, and reply as its own event with its own timestamp.

## Core design principles

- event-level data
- depth cap of 3
- comment window of T0 to T0 + 60 days
- thread-first collection
- one-to-many firm mapping downstream
- Parquet as source of truth
- Excel/Sheets as convenience outputs

## Main pipeline

1. `scum_collect.py`
2. `scum_score.py`
3. `scum_aggregate.py`
