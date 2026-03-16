# SCUM2 Runbook

## Monthly workflow

### 1. Collect
Run `scum_collect.py` for the desired date range.

### 2. Score
Run `scum_score.py` on new raw events.

### 3. Aggregate
Run `scum_aggregate.py` to build daily SCUM outputs.

## Principles

- raw data lives in Drive
- GitHub stores code, prompts, config, and docs
- Parquet is the main storage format
- Excel and Sheets are review outputs
