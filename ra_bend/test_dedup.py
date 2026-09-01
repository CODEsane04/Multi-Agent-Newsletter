from agents.scout import fetch_rss_feeds
import json
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from graph.state import PipelineState

load_dotenv()


class OpSchema(BaseModel):
    to_remove_index: str = Field(
        description="The index to remove (e.g. '2' or '3'), or 'None' if the items are about different news."
    )
    rejection_reason: str = Field(
        description="Reason why the news item was rejected, or 'None'."
    )

def remove_duplicates(result) :
    news_items = result

    items_to_keep = []

    # ─── Step 1: Build text representations (Title + Snippet) ───
    texts = [f"{item['title']} {item['snippet']}" for item in news_items]

    # ─── Step 2: Convert snippets to vector embeddings in batches ───
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    batch_size = 8
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_vectors = embedding_model.embed_documents(batch)
        all_embeddings.extend(batch_vectors)

    # ─── Step 3: Cosine Similarity Matrix (L2 Normalized) ───
    vector_matrix = np.array(all_embeddings)
    norms = np.linalg.norm(vector_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized_matrix = vector_matrix / norms
    dot_product_matrix = np.dot(normalized_matrix, normalized_matrix.T)

    # ─── Step 4: Filter out duplicate snippets via LLM judge ───
    llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0.1)
    structured_model = llm.with_structured_output(
        schema=OpSchema.model_json_schema(),
        method="json_schema"
    )

    prompt = PromptTemplate(
        template="""You are a news deduplication judge. You are given a pair of news items with high similarity.
        Determine if they cover the EXACT SAME news event.

        Rules:
        1. If they cover DIFFERENT news events, return to_remove_index as "None".
        2. If they cover the SAME news event, choose the one with LESS information/clarity to remove, and return its index ({i} or {j}).

        Output JSON matching schema:
        {{"to_remove_index": "{i}" or "{j}" or "None", "rejection_reason": "explanation or None"}}

        Item {i}: {snip1}
        Item {j}: {snip2}""",
        input_variables=["snip1", "i", "snip2", "j"]
    )

    chain = prompt | structured_model
    similarity_threshold = 0.85
    to_remove = set()

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if i in to_remove or j in to_remove:
                continue

            if dot_product_matrix[i][j] > similarity_threshold:
                print("\n for this run \n")
                try:
                    response = chain.invoke({
                        "snip1": texts[i],
                        "i": str(i),
                        "snip2": texts[j],
                        "j": str(j)
                    })

                    print(type(response))

                    raw_idx = response.get("to_remove_index") if isinstance(response, dict) else getattr(response, "to_remove_index", None)
                    reason = response.get("rejection_reason") if isinstance(response, dict) else getattr(response, "rejection_reason", "Duplicate news")

                    if raw_idx and str(raw_idx).strip().lower() not in ["none", "null", ""]:
                        remove_idx = int(str(raw_idx).strip())
                        to_remove.add(remove_idx)
                        print(f"[Dedup] Removing item {remove_idx}. Reason: {reason}")

                except Exception as e:
                    print(f"[Dedup] LLM invocation failed for items {i} and {j}: {e}")

    # ─── Step 5: Keep non-duplicate items ───
    for idx, item in enumerate(news_items):
        if idx not in to_remove:
            item["status"] = "deduplicated"
            items_to_keep.append(item)

    total_removed = len(to_remove)
    

    print(f"[Dedup] Original: {len(news_items)} | Kept: {len(items_to_keep)} | Removed: {total_removed}")

    return items_to_keep

if __name__ == "__main__" :
    results = fetch_rss_feeds()
    items_to_keep = remove_duplicates(result=results)