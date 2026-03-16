# SCUM2 Expand Schema

SCUM2 expansion converts raw Reddit events into firm-specific candidate rows before scoring.

## Why this step exists

A single Reddit thread can mention multiple firms. SCUM2 therefore cannot score only at the thread level. It must first identify all relevant firm candidates, then score sentiment about each firm separately.

Example:

One thread mentions:
- Citadel
- Jane Street
- Hudson River Trading

That thread should produce three firm-specific candidates.

## Outputs

### 1. thread_firm_pairs

One row per thread x firm candidate.

Fields:

- `post_id`
- `canonical_name`
- `matched_aliases`
- `thread_direct_match`
- `thread_text`
- `thread_title`
- `subreddit`
- `thread_created_utc`
- `candidate_strength`

### 2. event_firm_pairs

One row per event x firm candidate.

Fields:

- `event_id`
- `post_id`
- `comment_id`
- `parent_id`
- `canonical_name`
- `matched_aliases`
- `candidate_source`
- `event_text`
- `created_utc`
- `depth`
- `score`
- `subreddit`

## Candidate source logic

Candidate rows may be created because of:

- direct alias match in the event text
- direct alias match elsewhere in the thread
- inherited thread context from the post or parent event

## Notes

This step creates candidates only. It does not assign final sentiment.
Scoring happens later in `scum_score.py`.
