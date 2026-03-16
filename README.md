# ci2-scum2

Time-aware Reddit reputation pipeline for hedge funds.

## What SCUM2 is

SCUM2 is the Reddit layer of CI2. It captures Reddit posts, comments, and replies as timestamped events, scores sentiment and confidence at the event level, and aggregates those into daily fund-level reputation series.

Unlike SCUM1, SCUM2 does not collapse all discussion onto the original post date. It treats Reddit as a stream of dated events.

## Core ideas

- event-level Reddit data
- thread-first collection
- one-to-many firm mapping
- time-correct sentiment placement
- Parquet as source of truth
- Excel/Sheets as convenience outputs

## Repo structure

- `config/` → firms, paths, run settings
- `prompts/` → scoring and relevance prompts
- `docs/` → schema, brief, runbook
- `scripts/` → collect, score, aggregate
- raw data lives in Google Drive, not in GitHub

## Main scripts

- `scripts/scum_collect.py`
- `scripts/scum_score.py`
- `scripts/scum_aggregate.py`

## Workflow

1. Collect Reddit events
2. Score relevant events
3. Aggregate to daily SCUM series
4. Export review files and rollups

## Status

Initial repo scaffold for SCUM2 build.
