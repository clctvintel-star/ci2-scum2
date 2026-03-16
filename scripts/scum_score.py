import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import anthropic
import pandas as pd
import requests
import trafilatura
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from newspaper import Article


# =========================================================
# CLI
# =========================================================

parser = argparse.ArgumentParser(description="CI2 QWASS enricher")
parser.add_argument("--input", type=str, required=True, help="Path to collector append CSV")
parser.add_argument("--output-dir", type=str, default=None, help="Directory for enriched outputs")
parser.add_argument("--sleep-seconds", type=float, default=0.8, help="Delay between network calls")
parser.add_argument("--llm-sleep-seconds", type=float, default=0.25, help="Delay between relevance calls")
parser.add_argument("--max-rows", type=int, default=None, help="Optional max rows for testing")
args = parser.parse_args()


# =========================================================
# REPO / CONFIG
# =========================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "prompts"


def load_yaml(path: str) -> dict:
    with open(REPO_ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8")


PATHS = load_yaml("config/paths.yaml")
FIRMS_CONFIG = load_yaml("config/firms.yaml")

RELEVANCE_PROMPT_PATH = PROMPTS_DIR / "relevance_gate_prompt.txt"
if RELEVANCE_PROMPT_PATH.exists():
    RELEVANCE_PROMPT_TEMPLATE = load_text(RELEVANCE_PROMPT_PATH)
else:
    RELEVANCE_PROMPT_TEMPLATE = """
You are deciding whether an article is relevant to a tracked hedge fund / trading firm.

Tracked firm: {fund_name}
Known aliases: {aliases}

Title: {title}
Snippet: {snippet}
Source: {source}
URL: {url}

Rules:
- KEEP if the tracked firm is directly mentioned, even briefly.
- KEEP if the story is about someone joining from, leaving from, being hired from, or being poached from the tracked firm.
- KEEP if the tracked firm is compared to peers on performance, talent, strategy, fees, culture, regulation, litigation, technology, crypto, AI, market making, or recruiting.
- KEEP if the tracked firm is mentioned as a former employer and that fact is meaningful to the story.
- KEEP if there is any plausible reputational relevance to allocators, job candidates, reporters, PR/comms, or rivals.
- DROP only if the article is clearly not about the tracked firm at all.
- When uncertain, choose KEEP or UNCERTAIN, not DROP.

Return exactly:

Decision: KEEP or DROP or UNCERTAIN
Confidence: <0.0 to 1.0>
Reason: <brief reason>
""".strip()

DRIVE_ROOT = PATHS["ci2"]["drive_root"]
QWASS_DB = PATHS["projects"]["qwass2"]["db"]

if "keys" in PATHS and "env_file" in PATHS["keys"]:
    ENV_FILE_REL = PATHS["keys"]["env_file"]
elif "paths" in PATHS and "keys_env" in PATHS["paths"]:
    ENV_FILE_REL = PATHS["paths"]["keys_env"]
else:
    ENV_FILE_REL = "ci2_keys.env"

ENV_PATH = (
    Path(DRIVE_ROOT) / ENV_FILE_REL
    if not str(ENV_FILE_REL).startswith("/content/")
    else Path(ENV_FILE_REL)
)

INPUT_PATH = Path(args.input)
STAMP = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")

DEFAULT_OUTPUT_DIR = Path(args.output_dir) if args.output_dir else (Path(DRIVE_ROOT) / QWASS_DB)
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENRICHED_PATH = DEFAULT_OUTPUT_DIR / f"collector_enriched_{STAMP}.csv"
MANUAL_QUEUE_PATH = DEFAULT_OUTPUT_DIR / f"collector_manual_queue_{STAMP}.csv"
REPORT_PATH = DEFAULT_OUTPUT_DIR / f"collector_enrich_report_{STAMP}.json"

CANONICAL_INPUT_COLUMNS = [
    "article_id",
    "date",
    "time",
    "utc",
    "title",
    "url",
    "normalized_url",
    "source",
    "author1",
    "author2",
    "summary",
    "summary_source",
    "retrieved_snippet",
    "snippet_engine",
    "fund_name",
    "collected_at",
    "query_text",
    "query_window_start",
    "query_window_end",
    "was_updated",
]

NEW_COLUMNS = [
    "relevance_decision",
    "relevance_confidence",
    "relevance_reason",
    "enrich_status",
    "word_count",
    "full_text_source",
    "boilerplate_stripped",
    "manual_review_flag",
    "archive_snapshot_url",
]

OUTPUT_COLUMNS = CANONICAL_INPUT_COLUMNS + NEW_COLUMNS

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://news.google.com/",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

MIN_WORDS_CLEAR_FAILURE = 150
MIN_WORDS_PARTIAL = 350
MIN_WORDS_STRONG_SUCCESS = 350

RELEVANCE_MODEL = "claude-haiku-4-5"
RELEVANCE_MAX_TOKENS = 220

CLAUDE_RETRY_ATTEMPTS = 3
REQUEST_TIMEOUT = 30

EXTRACTOR_RETRIES = 3
EXTRACTOR_SLEEP_SECONDS = 1.2
MIN_RETRY_ACCEPT_WORDS = 20

# Very conservative drop threshold
DROP_CONFIDENCE_THRESHOLD = 0.93

ARCHIVE_MIRRORS = [
    "archive.ph",
    "archive.is",
    "archive.today",
    "archive.fo",
    "archive.li",
    "archive.md",
    "archive.vn",
]

PAYWALL_PRIORITY_DOMAINS = [
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "barrons.com",
    "economist.com",
    "markets.businessinsider.com",
    "businessinsider.com",
]

MANUAL_REVIEW_DOMAINS = [
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "barrons.com",
    "economist.com",
]

BOILERPLATE_PATTERNS = [
    r"subscribe now.*",
    r"sign up for.*newsletter.*",
    r"share this article.*",
    r"follow us on .*",
    r"all rights reserved.*",
    r"copyright\s+\d{4}.*",
    r"advertisement",
    r"recommended stories.*",
    r"read more:.*",
    r"gift this article.*",
    r"save article.*",
    r"listen to this article.*",
    r"create a free account.*",
    r"already have an account\?.*",
    r"to continue reading.*",
    r"register now.*",
    r"unlock this article.*",
    r"explore more offers.*",
    r"terms & conditions apply.*",
    r"before it.?s here, it.?s on the bloomberg terminal.*",
    r"make sense of the markets.*",
    r"jump to comments.*",
    r"sign in.*",
    r"skip to content.*",
    r"share on facebook.*",
    r"share on twitter.*",
    r"copy link.*",
    r"this story has been shared.*",
]

LOW_QUALITY_SOURCE_PENALTIES = {
    "raw_html_text": 40,
    "wayback_raw_html_text": 50,
    "archive_raw_html_text": 50,
    "trafilatura_url": 5,
    "wayback_trafilatura_url": 12,
    "archive_trafilatura_url": 12,
}

SOURCE_BONUSES = {
    "trafilatura_html": 10,
    "newspaper_html": 7,
    "json_ld_articlebody": 9,
    "archive_trafilatura_html": 14,
    "archive_newspaper_html": 12,
    "archive_json_ld_articlebody": 10,
    "wayback_trafilatura_html": 8,
    "wayback_newspaper_html": 6,
    "trafilatura_url": 4,
    "newspaper_url": 2,
}


# =========================================================
# ENV / CLIENTS
# =========================================================

def setup_env() -> anthropic.Anthropic:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"Missing env file: {ENV_PATH}")

    load_dotenv(ENV_PATH)

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError(f"ANTHROPIC_API_KEY not found in {ENV_PATH}")

    return anthropic.Anthropic(api_key=anthropic_api_key)


# =========================================================
# FIRM / ALIAS HELPERS
# =========================================================

def normalize_text(s: str) -> str:
    s = str(s or "").lower().strip()
    s = s.replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"[^a-z0-9&.\-'/ ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def get_aliases_for_firm(fund_name: str) -> List[str]:
    definitions = FIRMS_CONFIG.get("firm_definitions", {})
    meta = definitions.get(fund_name, {})
    aliases = [fund_name] + list(meta.get("aliases_safe", []))
    aliases = [str(a).strip() for a in aliases if str(a).strip()]

    deduped = []
    seen = set()
    for alias in aliases:
        key = normalize_text(alias)
        if key not in seen:
            seen.add(key)
            deduped.append(alias)

    return deduped


def alias_hit(text: str, aliases: List[str]) -> bool:
    norm_text = normalize_text(text)
    if not norm_text:
        return False

    for alias in aliases:
        a = normalize_text(alias)
        if not a:
            continue
        if a in norm_text:
            return True
    return False


# =========================================================
# LOAD / WORD COUNTS
# =========================================================

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in CANONICAL_INPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    for col in NEW_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def word_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\S+\b", text))


def should_trust_existing_text(summary_text: str, summary_source: str) -> bool:
    wc = word_count(summary_text)
    src = str(summary_source or "").strip().lower()

    if wc >= MIN_WORDS_STRONG_SUCCESS and src not in {"google_news", ""}:
        return True
    if wc >= 600:
        return True
    return False


def classify_text_length(wc: int) -> str:
    if wc < MIN_WORDS_CLEAR_FAILURE:
        return "too_short"
    if wc < MIN_WORDS_PARTIAL:
        return "partial"
    return "strong"


# =========================================================
# CLAUDE RELEVANCE GATE
# =========================================================

def parse_relevance_response(text: str) -> Tuple[str, float, str]:
    decision = "UNCERTAIN"
    confidence = 0.5
    reason = "[no reason]"

    if not text:
        return decision, confidence, reason

    d_match = re.search(r"Decision:\s*(KEEP|DROP|UNCERTAIN)", text, flags=re.IGNORECASE)
    c_match = re.search(r"Confidence:\s*([-+]?\d*\.?\d+)", text, flags=re.IGNORECASE)
    r_match = re.search(r"Reason:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)

    if d_match:
        decision = d_match.group(1).upper()
    if c_match:
        confidence = float(c_match.group(1))
        confidence = max(0.0, min(1.0, confidence))
    if r_match:
        reason = r_match.group(1).strip()

    return decision, confidence, reason


def build_relevance_prompt(
    fund_name: str,
    aliases: List[str],
    title: str,
    snippet: str,
    source: str,
    url: str,
) -> str:
    alias_text = ", ".join(aliases)
    return RELEVANCE_PROMPT_TEMPLATE.format(
        fund_name=fund_name,
        aliases=alias_text,
        title=title,
        snippet=snippet,
        source=source,
        url=url,
    )


def call_claude_relevance(
    client: anthropic.Anthropic,
    fund_name: str,
    aliases: List[str],
    title: str,
    snippet: str,
    source: str,
    url: str,
) -> Tuple[str, float, str]:
    prompt = build_relevance_prompt(
        fund_name=fund_name,
        aliases=aliases,
        title=title,
        snippet=snippet,
        source=source,
        url=url,
    )

    last_error = None
    for attempt in range(1, CLAUDE_RETRY_ATTEMPTS + 1):
        try:
            response = client.messages.create(
                model=RELEVANCE_MODEL,
                max_tokens=RELEVANCE_MAX_TOKENS,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = []
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            text = "\n".join(parts).strip()
            return parse_relevance_response(text)
        except Exception as e:
            last_error = e
            print(
                f"⚠️ Relevance gate attempt {attempt}/{CLAUDE_RETRY_ATTEMPTS} failed: {e}",
                flush=True,
            )
            time.sleep(2.0)

    print(f"⚠️ Relevance gate fallback due to repeated failure: {last_error}", flush=True)
    return "UNCERTAIN", 0.3, "Model call failed; defaulting to uncertain."


# =========================================================
# FETCH / EXTRACTION HELPERS
# =========================================================

def get_domain(url: str) -> str:
    m = re.search(r"https?://([^/]+)", str(url).strip().lower())
    return m.group(1).replace("www.", "") if m else ""


def fetch_html(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if resp.status_code == 200 and resp.text:
            return resp.text
    except Exception:
        return None
    return None


def resolve_redirect(url: str) -> str:
    if not url:
        return url

    for method in ("head", "get"):
        try:
            if method == "head":
                resp = requests.head(url, headers=HEADERS, timeout=12, allow_redirects=True)
            else:
                resp = requests.get(url, headers=HEADERS, timeout=12, allow_redirects=True, stream=True)

            final_url = str(getattr(resp, "url", "") or "").strip()
            if final_url:
                return final_url
        except Exception:
            continue

    return url


def trafilatura_extract_from_url(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_links=False,
            favor_precision=True,
            deduplicate=True,
        )
        return (text or "").strip()
    except Exception:
        return ""


def trafilatura_extract_from_html(html_text: str) -> str:
    try:
        text = trafilatura.extract(
            html_text,
            include_comments=False,
            include_links=False,
            favor_precision=True,
            deduplicate=True,
        )
        return (text or "").strip()
    except Exception:
        return ""


def newspaper_extract(url: str, html_text: Optional[str] = None) -> str:
    try:
        art = Article(url, language="en")
        if html_text:
            art.set_html(html_text)
            art.parse()
        else:
            art.download()
            art.parse()
        return (art.text or "").strip()
    except Exception:
        return ""


def wayback_url(url: str) -> str:
    return f"https://web.archive.org/web/0/{quote(url, safe=':/?&=%')}"


def extract_meta_refresh_target(html_text: str) -> Optional[str]:
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        meta = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
        if not meta:
            return None
        content = meta.get("content", "")
        m = re.search(r'url=(.+)$', content, flags=re.I)
        if not m:
            return None
        return m.group(1).strip().strip("'").strip('"')
    except Exception:
        return None


def find_archive_snapshot_in_html(html_text: str) -> Optional[str]:
    if not html_text:
        return None

    patterns = [
        r'https?://archive\.(?:ph|is|today|fo|li|md|vn)/[A-Za-z0-9]+',
        r'https?://archive\.(?:ph|is|today|fo|li|md|vn)/o/[A-Za-z0-9]+',
    ]

    for pattern in patterns:
        m = re.search(pattern, html_text)
        if m:
            return m.group(0)

    try:
        soup = BeautifulSoup(html_text, "html.parser")

        # canonical
        for tag in soup.find_all("link", rel=True):
            rels = " ".join(tag.get("rel", [])).lower()
            href = tag.get("href", "") or ""
            if "canonical" in rels and re.search(r"https?://archive\.(?:ph|is|today|fo|li|md|vn)/", href):
                return href.strip()

        # og:url
        for tag in soup.find_all("meta"):
            prop = (tag.get("property") or "").lower()
            content = tag.get("content") or ""
            if prop == "og:url" and re.search(r"https?://archive\.(?:ph|is|today|fo|li|md|vn)/", content):
                return content.strip()

        # first likely snapshot link
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if re.search(r"https?://archive\.(?:ph|is|today|fo|li|md|vn)/[A-Za-z0-9]+", href):
                return href
    except Exception:
        pass

    return None


def lookup_archive_snapshot(url: str) -> Optional[str]:
    original = url.strip()
    if not original:
        return None

    for domain in ARCHIVE_MIRRORS:
        candidate_urls = [
            f"https://{domain}/{original}",
            f"https://{domain}/?url={quote(original, safe='')}",
            f"https://{domain}/?run=1&url={quote(original, safe='')}",
        ]

        for lookup_url in candidate_urls:
            try:
                resp = requests.get(lookup_url, headers=HEADERS, timeout=18, allow_redirects=True)
                final_url = str(getattr(resp, "url", "") or "").strip()
                html = resp.text or ""

                if re.fullmatch(rf"https://{re.escape(domain)}/[A-Za-z0-9]+/?", final_url):
                    return final_url.rstrip("/")

                meta_target = extract_meta_refresh_target(html)
                if meta_target and re.search(r"https?://archive\.(?:ph|is|today|fo|li|md|vn)/", meta_target):
                    return meta_target.rstrip("/")

                found = find_archive_snapshot_in_html(html)
                if found:
                    return found.rstrip("/")
            except Exception:
                continue

    return None


# =========================================================
# CLEANUP / EXTRA TEXT SOURCES
# =========================================================

def clean_html_to_text(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_json_ld_articlebody(html_text: str) -> str:
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        bodies = []

        for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = tag.string or tag.get_text() or ""
            raw = raw.strip()
            if not raw:
                continue

            try:
                data = json.loads(raw)
            except Exception:
                continue

            stack = [data]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    article_body = item.get("articleBody")
                    if isinstance(article_body, str) and article_body.strip():
                        bodies.append(article_body.strip())
                    for value in item.values():
                        if isinstance(value, (dict, list)):
                            stack.append(value)
                elif isinstance(item, list):
                    stack.extend(item)

        if not bodies:
            return ""

        best = max(bodies, key=word_count)
        return best.strip()
    except Exception:
        return ""


def strip_boilerplate(text: str) -> Tuple[str, bool]:
    if not text:
        return "", False

    original = text
    cleaned = text

    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    lines = [ln.strip() for ln in cleaned.splitlines()]
    filtered = []
    for ln in lines:
        if not ln:
            continue
        low = ln.lower()
        if len(ln.split()) <= 3 and low in {
            "share", "subscribe", "advertisement", "newsletter", "sign in", "read more"
        }:
            continue
        filtered.append(ln)

    cleaned = "\n".join(filtered)
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()

    changed = cleaned != original
    return cleaned, changed


# =========================================================
# EXTRACTION CORE
# =========================================================

def retry_extract(func, *args):
    last_text = ""
    for attempt in range(EXTRACTOR_RETRIES):
        try:
            text = func(*args)
            if text and word_count(text) >= MIN_RETRY_ACCEPT_WORDS:
                return text
            last_text = text or ""
        except Exception:
            pass
        time.sleep(EXTRACTOR_SLEEP_SECONDS * (attempt + 1))
    return last_text or ""


def add_candidate(candidates: List[Dict], text: str, source_label: str):
    if not text:
        return

    cleaned, changed = strip_boilerplate(text)
    wc = word_count(cleaned)

    if wc == 0:
        return

    line_count = len([ln for ln in cleaned.splitlines() if ln.strip()])
    avg_line_words = (wc / line_count) if line_count else wc
    short_line_penalty = 0

    if avg_line_words < 5:
        short_line_penalty += 45
    elif avg_line_words < 8:
        short_line_penalty += 20

    source_penalty = LOW_QUALITY_SOURCE_PENALTIES.get(source_label, 0)
    source_bonus = SOURCE_BONUSES.get(source_label, 0)

    score = wc - source_penalty - short_line_penalty + source_bonus

    candidates.append({
        "text": cleaned,
        "source": source_label,
        "word_count": wc,
        "boilerplate_stripped": changed,
        "score": score,
    })


def choose_best_candidate(candidates: List[Dict]) -> Tuple[str, str, bool]:
    if not candidates:
        return "", "", False

    candidates = sorted(
        candidates,
        key=lambda x: (x["score"], x["word_count"]),
        reverse=True
    )
    best = candidates[0]
    return best["text"], best["source"], bool(best["boilerplate_stripped"])


def try_extract_full_text(url: str) -> Tuple[str, str, bool, Optional[str]]:
    candidates: List[Dict] = []

    canonical_url = resolve_redirect(url)
    domain = get_domain(canonical_url)
    archive_snapshot_url = None

    # 1) HTML-first on canonical URL
    html_text = retry_extract(fetch_html, canonical_url)
    if html_text:
        add_candidate(candidates, retry_extract(trafilatura_extract_from_html, html_text), "trafilatura_html")
        add_candidate(candidates, retry_extract(newspaper_extract, canonical_url, html_text), "newspaper_html")
        add_candidate(candidates, retry_extract(extract_json_ld_articlebody, html_text), "json_ld_articlebody")
        add_candidate(candidates, clean_html_to_text(html_text), "raw_html_text")

    # 2) URL-based extraction on canonical URL
    add_candidate(candidates, retry_extract(trafilatura_extract_from_url, canonical_url), "trafilatura_url")
    add_candidate(candidates, retry_extract(newspaper_extract, canonical_url), "newspaper_url")

    # 3) archive fallback BEFORE wayback for paywalled / hard domains
    archive_first = domain in PAYWALL_PRIORITY_DOMAINS

    if archive_first:
        archive_snapshot_url = lookup_archive_snapshot(canonical_url)
        if archive_snapshot_url:
            archive_html = retry_extract(fetch_html, archive_snapshot_url)
            if archive_html:
                add_candidate(candidates, retry_extract(trafilatura_extract_from_html, archive_html), "archive_trafilatura_html")
                add_candidate(candidates, retry_extract(newspaper_extract, archive_snapshot_url, archive_html), "archive_newspaper_html")
                add_candidate(candidates, retry_extract(extract_json_ld_articlebody, archive_html), "archive_json_ld_articlebody")
                add_candidate(candidates, clean_html_to_text(archive_html), "archive_raw_html_text")
            add_candidate(candidates, retry_extract(trafilatura_extract_from_url, archive_snapshot_url), "archive_trafilatura_url")

    # 4) Wayback fallback
    wb_url = wayback_url(canonical_url)
    wb_html = retry_extract(fetch_html, wb_url)
    if wb_html:
        add_candidate(candidates, retry_extract(trafilatura_extract_from_html, wb_html), "wayback_trafilatura_html")
        add_candidate(candidates, retry_extract(newspaper_extract, wb_url, wb_html), "wayback_newspaper_html")
        add_candidate(candidates, retry_extract(extract_json_ld_articlebody, wb_html), "wayback_json_ld_articlebody")
        add_candidate(candidates, clean_html_to_text(wb_html), "wayback_raw_html_text")
    add_candidate(candidates, retry_extract(trafilatura_extract_from_url, wb_url), "wayback_trafilatura_url")

    # 5) archive fallback after wayback for everyone else
    if not archive_first:
        archive_snapshot_url = lookup_archive_snapshot(canonical_url)
        if archive_snapshot_url:
            archive_html = retry_extract(fetch_html, archive_snapshot_url)
            if archive_html:
                add_candidate(candidates, retry_extract(trafilatura_extract_from_html, archive_html), "archive_trafilatura_html")
                add_candidate(candidates, retry_extract(newspaper_extract, archive_snapshot_url, archive_html), "archive_newspaper_html")
                add_candidate(candidates, retry_extract(extract_json_ld_articlebody, archive_html), "archive_json_ld_articlebody")
                add_candidate(candidates, clean_html_to_text(archive_html), "archive_raw_html_text")
            add_candidate(candidates, retry_extract(trafilatura_extract_from_url, archive_snapshot_url), "archive_trafilatura_url")

    best_text, best_source, best_stripped = choose_best_candidate(candidates)
    return best_text, best_source, best_stripped, archive_snapshot_url


# =========================================================
# OUTPUT HELPERS
# =========================================================

def make_manual_row(out: Dict, domain: str) -> Dict:
    return {
        "article_id": out.get("article_id", ""),
        "fund_name": out.get("fund_name", ""),
        "title": out.get("title", ""),
        "url": out.get("url", ""),
        "source": out.get("source", ""),
        "summary": out.get("summary", ""),
        "relevance_decision": out.get("relevance_decision", ""),
        "relevance_confidence": out.get("relevance_confidence", ""),
        "relevance_reason": out.get("relevance_reason", ""),
        "enrich_status": out.get("enrich_status", ""),
        "domain": domain,
        "hard_domain_flag": domain in MANUAL_REVIEW_DOMAINS,
        "word_count": out.get("word_count", 0),
        "full_text_source": out.get("full_text_source", ""),
        "archive_snapshot_url": out.get("archive_snapshot_url", ""),
    }


# =========================================================
# MAIN
# =========================================================

def main():
    client = setup_env()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    df = ensure_columns(df)

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    rows_loaded = len(df)
    print(f"✅ Loaded append rows: {rows_loaded}", flush=True)
    print(f"Input:  {INPUT_PATH}", flush=True)
    print(f"Output: {ENRICHED_PATH}", flush=True)

    enriched_rows: List[Dict] = []
    manual_rows: List[Dict] = []

    report = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "input_file": str(INPUT_PATH),
        "enriched_file": str(ENRICHED_PATH),
        "manual_queue_file": str(MANUAL_QUEUE_PATH),
        "rows_loaded": rows_loaded,
        "relevance_counts": {"KEEP": 0, "DROP": 0, "UNCERTAIN": 0},
        "status_counts": {},
    }

    for idx, row in df.iterrows():
        out = row.to_dict()

        fund_name = str(out.get("fund_name", "")).strip()
        aliases = get_aliases_for_firm(fund_name)

        title = str(out.get("title", "") or "").strip()
        snippet = str(out.get("summary", "") or "").strip()
        source = str(out.get("source", "") or "").strip()
        url = str(out.get("url", "") or "").strip()
        summary_source = str(out.get("summary_source", "") or "").strip()

        existing_wc = word_count(snippet)
        domain = get_domain(url)

        decision, confidence, reason = call_claude_relevance(
            client=client,
            fund_name=fund_name,
            aliases=aliases,
            title=title,
            snippet=snippet,
            source=source,
            url=url,
        )

        # HARD SAFETY OVERRIDE: never drop if any alias appears in title/snippet
        if decision == "DROP" and alias_hit(f"{title}\n{snippet}", aliases):
            decision = "KEEP"
            confidence = max(confidence, 0.8)
            reason = f"Safety override: alias hit in title/snippet. Original reason: {reason}"

        out["relevance_decision"] = decision
        out["relevance_confidence"] = confidence
        out["relevance_reason"] = reason

        report["relevance_counts"][decision] = report["relevance_counts"].get(decision, 0) + 1

        # ONLY drop on explicit high-confidence DROP with no alias evidence
        if decision == "DROP" and confidence >= DROP_CONFIDENCE_THRESHOLD:
            out["enrich_status"] = "dropped_relevance_gate_high_confidence"
            out["word_count"] = existing_wc
            out["full_text_source"] = ""
            out["boilerplate_stripped"] = False
            out["manual_review_flag"] = False
            out["archive_snapshot_url"] = ""
            enriched_rows.append(out)
            print(f"🗑️ DROP [{idx+1}/{rows_loaded}] {title[:90]}", flush=True)
            time.sleep(args.llm_sleep_seconds)
            continue

        # KEEP or UNCERTAIN or low-confidence DROP => continue
        if should_trust_existing_text(snippet, summary_source):
            cleaned, changed = strip_boilerplate(snippet)
            out["summary"] = cleaned
            out["summary_source"] = out.get("summary_source") or "existing"
            out["retrieved_snippet"] = out.get("retrieved_snippet") or ""
            out["snippet_engine"] = out.get("snippet_engine") or ""
            out["was_updated"] = bool(changed)
            out["enrich_status"] = "kept_existing_long_text"
            out["word_count"] = word_count(cleaned)
            out["full_text_source"] = "existing_summary"
            out["boilerplate_stripped"] = changed
            out["manual_review_flag"] = False
            out["archive_snapshot_url"] = ""
            enriched_rows.append(out)
            print(f"✅ KEEP EXISTING [{idx+1}/{rows_loaded}] wc={out['word_count']} | {title[:90]}", flush=True)
            time.sleep(args.llm_sleep_seconds)
            continue

        text, source_label, stripped_flag, archive_snapshot_url = try_extract_full_text(url)
        out["archive_snapshot_url"] = archive_snapshot_url or ""

        # SECOND SAFETY OVERRIDE: if extracted text contains alias, never drop
        if text and decision == "DROP" and alias_hit(text, aliases):
            out["relevance_decision"] = "KEEP"
            out["relevance_confidence"] = max(confidence, 0.8)
            out["relevance_reason"] = (
                f"Safety override: alias hit in extracted text. Original reason: {reason}"
            )

        if text:
            cleaned_wc = word_count(text)
            strength = classify_text_length(cleaned_wc)

            out["summary"] = text
            out["summary_source"] = source_label
            out["retrieved_snippet"] = ""
            out["snippet_engine"] = source_label
            out["was_updated"] = True
            out["word_count"] = cleaned_wc
            out["full_text_source"] = source_label
            out["boilerplate_stripped"] = stripped_flag

            if strength == "strong":
                out["enrich_status"] = f"success_{source_label}"
                out["manual_review_flag"] = False
                enriched_rows.append(out)
                print(f"✅ ENRICHED [{idx+1}/{rows_loaded}] {source_label} wc={cleaned_wc} | {title[:90]}", flush=True)
            else:
                out["enrich_status"] = f"manual_review_needed_{source_label}_{strength}"
                out["manual_review_flag"] = True
                enriched_rows.append(out)
                manual_rows.append(make_manual_row(out, domain))
                print(f"⚠️ PARTIAL [{idx+1}/{rows_loaded}] {source_label} wc={cleaned_wc} | {title[:90]}", flush=True)
        else:
            out["enrich_status"] = "manual_review_needed_failed_extraction"
            out["word_count"] = existing_wc
            out["full_text_source"] = ""
            out["boilerplate_stripped"] = False
            out["manual_review_flag"] = True

            if not snippet:
                out["summary"] = ""

            enriched_rows.append(out)
            manual_rows.append(make_manual_row(out, domain))
            print(f"⚠️ MANUAL [{idx+1}/{rows_loaded}] {domain} | {title[:90]}", flush=True)

        time.sleep(args.sleep_seconds)

    enriched_df = pd.DataFrame(enriched_rows)
    enriched_df = ensure_columns(enriched_df)
    enriched_df = enriched_df.reindex(columns=OUTPUT_COLUMNS)

    manual_df = pd.DataFrame(manual_rows)

    status_counts = enriched_df["enrich_status"].fillna("").astype(str).value_counts().to_dict()
    report["status_counts"] = status_counts
    report["manual_queue_count"] = len(manual_df)

    enriched_df.to_csv(ENRICHED_PATH, index=False)

    if len(manual_df) > 0:
        manual_df.to_csv(MANUAL_QUEUE_PATH, index=False)
    else:
        pd.DataFrame(columns=[
            "article_id",
            "fund_name",
            "title",
            "url",
            "source",
            "summary",
            "relevance_decision",
            "relevance_confidence",
            "relevance_reason",
            "enrich_status",
            "domain",
            "hard_domain_flag",
            "word_count",
            "full_text_source",
            "archive_snapshot_url",
        ]).to_csv(MANUAL_QUEUE_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== ENRICHMENT SUMMARY ===", flush=True)
    print(f"Rows loaded:          {rows_loaded}", flush=True)
    print(f"KEEP:                 {report['relevance_counts'].get('KEEP', 0)}", flush=True)
    print(f"DROP:                 {report['relevance_counts'].get('DROP', 0)}", flush=True)
    print(f"UNCERTAIN:            {report['relevance_counts'].get('UNCERTAIN', 0)}", flush=True)
    print(f"Manual queue count:   {len(manual_df)}", flush=True)
    print(f"Saved enriched file:  {ENRICHED_PATH}", flush=True)
    print(f"Saved manual queue:   {MANUAL_QUEUE_PATH}", flush=True)
    print(f"Saved report:         {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
