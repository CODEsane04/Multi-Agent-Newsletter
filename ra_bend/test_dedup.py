"""
Test: Deduplication via Agglomerative Clustering + Gemma LLM Judge
1. Fetch all news items from Scout
2. Embed title + snippet using TF-IDF (fully local, no downloads)
3. Run Agglomerative Clustering (cosine distance threshold)
4. For multi-item clusters, send to Gemma for pairwise comparison
5. Return deduplicated list + count of removed items
"""

import sys
import json

sys.path.insert(0, ".")

from agents.scout import fetch_rss_feeds
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import numpy as np

load_dotenv()

# ─── Step 1: Fetch all news items ───
print("Fetching news items...")
items = fetch_rss_feeds()
print(f"Fetched {len(items)} items\n")

# ─── Step 2: Embed title + snippet using TF-IDF ───
print("Generating TF-IDF vectors...")
texts = [f"{item['title']} {item['snippet']}" for item in items]

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(texts)
embeddings_array = tfidf_matrix.toarray()

# Normalize for cosine distance
norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
norms[norms == 0] = 1  # Avoid division by zero
embeddings_normalized = embeddings_array / norms

print(f"Generated vectors for {len(texts)} items (dim: {embeddings_array.shape[1]})\n")

# ─── Step 3: Agglomerative Clustering ───
print("Running Agglomerative Clustering...")
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.5,  # TF-IDF needs a higher threshold than neural embeddings
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

print(f"Total clusters: {len(clusters)}")
print(f"Unique items (single-item clusters): {len(single_clusters)}")
print(f"Potential duplicate groups: {len(multi_clusters)}\n")

# ─── Step 4: LLM Judge for multi-item clusters ───
items_to_keep = []

# Keep all single-cluster items directly
for cluster_id, indices in single_clusters.items():
    items_to_keep.append(items[indices[0]])

# For multi-item clusters, ask Gemma to judge
if multi_clusters:
    print("Sending duplicate candidates to Gemma for judgment...\n")

    llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0.1)
    parser = StrOutputParser()
    chain = llm | parser

    for cluster_id, indices in multi_clusters.items():
        cluster_items = [items[i] for i in indices]

        print(f"--- Cluster {cluster_id} ({len(cluster_items)} items) ---")
        for ci in cluster_items:
            print(f"  [{ci['source']}] {ci['title']}")

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
            print(f"  Gemma response: {result}\n")

            # Parse the JSON response
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            decision = json.loads(cleaned)
            keep_indices = decision.get("keep", list(range(1, len(cluster_items) + 1)))

            for idx in keep_indices:
                items_to_keep.append(cluster_items[idx - 1])  # Convert 1-based to 0-based

        except Exception as e:
            print(f"  Error parsing Gemma response: {e}")
            print(f"  Keeping all items in this cluster as fallback.\n")
            items_to_keep.extend(cluster_items)

else:
    print("No potential duplicates found — all items are unique.\n")

# ─── Step 5: Results ───
removed_count = len(items) - len(items_to_keep)

print(f"\n{'='*60}")
print(f"DEDUP RESULTS")
print(f"{'='*60}")
print(f"Original items:  {len(items)}")
print(f"After dedup:     {len(items_to_keep)}")
print(f"Removed:         {removed_count}")
print(f"\nFinal items:")
for item in items_to_keep:
    print(f"  [{item['source']}] {item['title']}")
