# Deduplication Layer — TF-IDF + Agglomerative Clustering + Gemma Judge

import json
from dotenv import load_dotenv
from graph.state import PipelineState
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import numpy as np

load_dotenv()


def remove_duplicates(state: PipelineState) -> PipelineState:
    news_items = state["items"]
    items_to_keep = []

    # ─── Step 1: Build text representations for vectorization ───
    texts = []
    for item in news_items:
        txt = f"{item['title']} {item['snippet']}"
        texts.append(txt)

    # ─── Step 2: Turn text to TF-IDF embeddings ───
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    embeddings_arr = tfidf_matrix.toarray()

    # Normalize for cosine distance
    norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_normalized = embeddings_arr / norms

    # ─── Step 3: Agglomerative Clustering ───
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.5,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings_normalized)

    # Group items by cluster
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    single_clusters = {k: v for k, v in clusters.items() if len(v) == 1}
    multi_clusters = {k: v for k, v in clusters.items() if len(v) > 1}

    # ─── Step 4: Keep all single-cluster items directly ───
    for cluster_id, indices in single_clusters.items():
        items_to_keep.append(news_items[indices[0]])

    # ─── Step 5: LLM Judge for multi-item clusters ───
    if multi_clusters:
        llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0.1)
        parser = StrOutputParser()
        chain = llm | parser

        for cluster_id, indices in multi_clusters.items():
            cluster_items = [news_items[i] for i in indices]

            # Build prompt for pairwise comparison
            items_description = ""
            for i, ci in enumerate(cluster_items):
                items_description += f"\nItem {i+1}:\n"
                items_description += f"  Title: {ci['title']}\n"
                items_description += f"  Source: {ci['source']}\n"
                items_description += f"  Snippet: {ci['snippet'][:200]}\n"

            prompt = f"""You are a news deduplication judge. Below are news items that appear similar.
Your job: Determine which items are about the EXACT SAME news event.

{items_description}

RULES:
- If items cover the SAME event/announcement, keep only the one with the richest, most informative content.
- If items are about DIFFERENT topics (just semantically similar), keep ALL of them.

Respond with ONLY a valid JSON object in this format:
{{"keep": [1, 3], "remove": [2], "reasoning": "Items 1 and 2 cover the same Anthropic announcement, Item 1 has more detail. Item 3 is about a different topic."}}

Use 1-based indexing matching the item numbers above."""

            try:
                result = chain.invoke(prompt)

                # Parse the JSON response (strip markdown code blocks if present)
                cleaned = result.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned.rsplit("```", 1)[0]
                cleaned = cleaned.strip()

                decision = json.loads(cleaned)
                keep_indices = decision.get("keep", list(range(1, len(cluster_items) + 1)))

                for idx in keep_indices:
                    items_to_keep.append(cluster_items[idx - 1])

            except Exception as e:
                print(f"[Dedup] Error parsing Gemma response: {e}")
                print(f"[Dedup] Keeping all items in cluster as fallback.")
                items_to_keep.extend(cluster_items)

    # ─── Step 6: Update status for all kept items ───
    for item in items_to_keep:
        item["status"] = "deduplicated"

    # ─── Step 7: Update PipelineState ───
    duplicates_removed = len(news_items) - len(items_to_keep)
    state["items"] = items_to_keep
    state["duplicates_removed"] = duplicates_removed

    print(f"[Dedup] Original: {len(news_items)} | After dedup: {len(items_to_keep)} | Removed: {duplicates_removed}")

    return state
