# SCUM2 Event Schema

## Raw events

Each Reddit post, comment, or reply is a separate row.

### Core fields

- `event_id`
- `post_id`
- `comment_id`
- `parent_id`
- `type`
- `subreddit`
- `created_utc`
- `depth`
- `author`
- `score`
- `title`
- `selftext`
- `body`
- `url`
- `permalink`

### Notes

- posts have `depth = 0`
- top-level comments have `depth = 1`
- replies have `depth = 2` or `depth = 3`
- raw events are collected thread-first and are not yet final fund-level analytic rows

## Scored events

Scored events extend raw events with:

- `canonical_name`
- `matched_alias`
- `mentions_fund`
- `relevance`
- `haiku_sentiment`
- `haiku_confidence`
- `haiku_wass`
- `gemini_sentiment`
- `gemini_confidence`
- `gemini_wass`
- `tiebreaker_sentiment`
- `tiebreaker_confidence`
- `tiebreaker_wass`
- `final_wass`
- `solomon_triggered`

## Aggregated outputs

### post_days
Grouped by:
- `canonical_name`
- `post_id`
- `day_offset`

Fields:
- `post_day_scum`
- `total_weight`
- `n_events`

### fund_days
Grouped by:
- `canonical_name`
- `date`

Fields:
- `scum`
- `ma28`
- `ma90`
- `total_weight`
- `n_posts`
- `n_events`
