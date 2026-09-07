# Sourcer Agent — Full content fetching (PDFs, articles, transcripts)
import re
import requests
import trafilatura
import xml.etree.ElementTree as ET
from typing import Optional
from graph.state import PipelineState
from typing import TypedDict

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
ARXIV_API_URL = "http://export.arxiv.org/api/query?id_list={paper_id}"

class DataSearchResults(TypedDict):
    model_name: Optional[str]
    parameters: Optional[str]
    context_window: Optional[str]
    mmlu_score: Optional[str]
    gsm8k_score: Optional[str]
    humaneval_score: Optional[str]
    api_cost_per_1m_tokens: Optional[str]
    source_url: Optional[str]
    sources : Optional[str]

class arxiv_schema(TypedDict) : 
    paper_author : Optional[str]
    paper_abstract : Optional[str]
    paper_conclusion : Optional[str]

class dist_schema(TypedDict):
    title : Optional[str]
    snippet : Optional[str]
    full_article : Optional[str]
    data_search_results : Optional[DataSearchResults]
    arxiv_data : Optional[arxiv_schema]


def extract_arxiv_id(text: str) -> Optional[str]:
    """Extract arxiv paper ID from article text (e.g., 2406.12345)."""
    patterns = [
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})',
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5})',
        r'arXiv:(\d{4}\.\d{4,5})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def fetch_arxiv_data(paper_id: str) -> Optional[dict]:
    """Fetch author and abstract from ArXiv API given a paper ID."""
    try:
        url = ARXIV_API_URL.format(paper_id=paper_id)
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None

        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        # Extract authors
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        author_str = ", ".join(authors) if authors else None

        # Extract abstract
        abstract_el = entry.find("atom:summary", ns)
        abstract = abstract_el.text.strip() if abstract_el is not None else None

        return {
            "paper_author": author_str,
            "paper_abstract": abstract,
            "paper_conclusion": None,  # Skipping for V1 (lives in PDF)
        }

    except Exception:
        return None


def make_distilled_context(state: PipelineState) -> PipelineState:
    news_items = state["items"]

    for item in news_items:
        news_class = item["news_class"]

        if news_class in [3, 4, 5, 6]:

            # fetch the full article here------
            try:
                resp = requests.get(
                    item["url"],
                    headers=HEADERS,
                    verify=False,
                    timeout=15,
                )
                if resp.status_code == 200:
                    full_text = trafilatura.extract(resp.text) or ""
                    print(f"\n[SOURCER] : len of full aricle is : {len(full_text)}]\n")
                else:
                    full_text = ""
                    item["error_log"].append(f"[Sourcer] HTTP {resp.status_code} for {item['url']}")
            except Exception as e:
                full_text = ""
                item["error_log"].append(f"[Sourcer] Failed to fetch article: {e}")
            # ----------------------------------

            item["distilled_context"] = {
                "title": item["title"],
                "snippet": item["snippet"],
                "full_article": full_text,
                "data_search_results": None,
                "arxiv_data": None,
            }
            item["status"] = "enriched"

        elif news_class == 2:
            # Class 2: Research papers — fetch article + try to get arxiv data
            try:
                resp = requests.get(
                    item["url"],
                    headers=HEADERS,
                    verify=False,
                    timeout=15,
                )
                if resp.status_code == 200:
                    full_text = trafilatura.extract(resp.text) or ""
                else:
                    full_text = ""
                    item["error_log"].append(f"[Sourcer] HTTP {resp.status_code} for {item['url']}")
            except Exception as e:
                full_text = ""
                item["error_log"].append(f"[Sourcer] Failed to fetch article: {e}")

            # Try to find and fetch arxiv paper data
            arxiv_data = None
            arxiv_id = extract_arxiv_id(full_text)
            if arxiv_id:
                arxiv_data = fetch_arxiv_data(arxiv_id)
                if arxiv_data:
                    print(f"[Sourcer] Found arxiv paper {arxiv_id} for: {item['title'][:50]}")

            item["distilled_context"] = {
                "title": item["title"],
                "snippet": item["snippet"],
                "full_article": full_text,
                "data_search_results": None,
                "arxiv_data": arxiv_data,
            }
            item["status"] = "enriched"

        elif news_class == 1:
            # Class 1 needs full article + data web search (handled separately)
            try:
                resp = requests.get(
                    item["url"],
                    headers=HEADERS,
                    verify=False,
                    timeout=15,
                )
                if resp.status_code == 200:
                    full_text = trafilatura.extract(resp.text) or ""
                else:
                    full_text = ""
                    item["error_log"].append(f"[Sourcer] HTTP {resp.status_code} for {item['url']}")
            except Exception as e:
                full_text = ""
                item["error_log"].append(f"[Sourcer] Failed to fetch article: {e}")

            item["distilled_context"] = {
                "title": item["title"],
                "snippet": item["snippet"],
                "full_article": full_text,
                "data_search_results": None,  # Will be filled by Data Web Search agent
                "arxiv_data": None,
            }
            item["status"] = "enriched"

    state["items"] = news_items
    print(f"[Sourcer] Enriched {len(news_items)} items with full article text")
    return state

