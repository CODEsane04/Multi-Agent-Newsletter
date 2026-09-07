# Scout Node — RSS fetching (Pure Python, no LLM)
import feedparser
import requests
from uuid import uuid4
from datetime import datetime, timezone

from graph.state import NewsItem, PipelineState

feeds = [
    ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("MIT Tech Review", "https://www.technologyreview.com/feed/"),
    ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ("Ars Technica AI", "https://feeds.arstechnica.com/arstechnica/technology-lab"),
]

MAX_ENTRIES_PER_FEED = 1
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_rss_feeds() -> list[NewsItem]:
    results = []

    for source_name, url in feeds:
        try:
            resp = requests.get(url, headers=HEADERS, verify=False, timeout=10)
            if resp.status_code != 200:
                print(f"[Scout] {source_name} returned HTTP {resp.status_code}, skipping.")
                continue

            parsed = feedparser.parse(resp.text)

            for entry in parsed.entries[:MAX_ENTRIES_PER_FEED]:
                item: NewsItem = {
                    # ─── Identity & Status ───
                    "item_id": str(uuid4()),
                    "status": "fetched",

                    # ─── Scout Output ───
                    "source": f"rss_{source_name.lower().replace(' ', '_')}",
                    "url": entry.get("link", ""),
                    "title": entry.get("title", ""),
                    "snippet": entry.get("summary", "")[:500],
                    "author": entry.get("author", "Unknown"),

                    # ─── Curator Output ───
                    "relevance_score": None,
                    "curator_reasoning": None,

                    # ─── Classifier Output ───
                    "news_class": None,
                    "classification_confidence": None,

                    # ─── Enrichment Output ───
                    "distilled_context": None,
                    "data_search_results": None,
                    "data_search_complete": False,

                    # ─── Writer Output ───
                    "writer_system_prompt": None,
                    "writer_draft": None,
                    "retry_count": 0,

                    # ─── Reviewer Output ───
                    "reviewer_status": None,
                    "reviewer_reasoning": None,
                    "hallucinations_found": [],
                    "formatting_errors": [],

                    # ─── Pipeline Control ───
                    "error_log": [],
                    "cumulative_tokens": 0,
                }
                results.append(item)

        except Exception as e:
            print(f"[Scout] Failed to fetch from {source_name}: {e}")
            continue

    return results

def scout_global_state_update(state : PipelineState) -> PipelineState:
    items = fetch_rss_feeds()

    state: PipelineState = {
        "run_id": str(uuid4()),
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
        'all_approved' : False,
        'iterations' : 0,
        'max_iterations' : 4,
        "total_signals_fetched": len(items),
        "duplicates_removed": 0,
        "items_filtered_out": 0,
        "total_cost_usd": 0.0,
        "run_status": "running",
    }

    return state