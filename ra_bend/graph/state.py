"""
State schemas for the Agentic AI Newsletter Pipeline.

NewsItem: Represents a single news item's full lifecycle.
PipelineState: Top-level run state — metadata + list of NewsItems.
"""

from typing import TypedDict, Optional, Literal


class NewsItem(TypedDict):
    """Single news item — its full lifecycle state from fetch to publish."""

    # ─── Identity & Status ───
    item_id: str
    status: Literal[
        "fetched",
        "deduplicated",
        "curated",
        "classified",
        "enriched",
        "drafted",
        "reviewing",
        "approved",
        "escalated",
        "published",
    ]

    # ─── Scout Output ───
    source: str  # "rss_techcrunch" | "rss_openai" | "rss_anthropic" | "reddit"
    url: str
    title: str
    snippet: str
    author: str

    # ─── Curator Output ───
    relevance_score: Optional[float]
    curator_reasoning: Optional[str]

    # ─── Classifier Output ───
    news_class: Optional[int]
    classification_confidence: Optional[float]

    # ─── Enrichment Output ───
    distilled_context: Optional[dict]
    data_search_results: Optional[dict]
    data_search_complete: bool

    # ─── Writer Output ───
    writer_system_prompt: Optional[str]
    writer_draft: Optional[str]
    retry_count: int

    # ─── Reviewer Output ───
    reviewer_status: Optional[Literal["APPROVED", "REJECTED"]]
    reviewer_reasoning: Optional[str]
    hallucinations_found: list[str]
    formatting_errors: list[str]

    # ─── Pipeline Control ───
    error_log: list[str]
    cumulative_tokens: int


class PipelineState(TypedDict):
    """Top-level state — run metadata + the list of all news items."""

    # Run Metadata
    run_id: str
    triggered_at: str

    # The List
    items: list[NewsItem]

    #loop metric
    all_approved : bool
    iterations : int
    max_iterations : int

    # Run-Level Metrics
    total_signals_fetched: int
    duplicates_removed: int
    items_filtered_out: int
    total_cost_usd: float
    run_status: Literal["running", "published", "failed"]
