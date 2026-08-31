# SLM Classifier Router — Batch classification via Gemma (batches of 8)

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from graph.state import PipelineState
from dotenv import load_dotenv

load_dotenv()

classes = [
    ("1", "Release of a new frontier model or an improved model"),
    ("2", "New Academic research or paper"),
    ("3", "New AI product or Application launch"),
    ("4", "Hardware or Infrastructure"),
    ("5", "Business policy and Geopolitics"),
    ("6", "General or miscellaneous"),
]

BATCH_CLASSIFICATION_PROMPT = """You are a news classifier. Classify each news item below into exactly one class:

Classes:
1. Release of a new frontier model or an improved model
2. New Academic research or paper
3. New AI product or Application launch
4. Hardware or Infrastructure
5. Business policy and Geopolitics
6. General or miscellaneous

News items:
{items_list}

Respond with ONLY a JSON array where each element is the class number (1-6) for the corresponding item, in order.
Example response for 3 items: [3, 5, 1]

Return ONLY the JSON array. No explanation."""

BATCH_SIZE = 8


def classify_news(state: PipelineState) -> PipelineState:
    news_items = state["items"]
    class_map = {int(k): v for k, v in classes}

    model = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0.1)
    parser = StrOutputParser()
    chain = model | parser

    # Process in batches of BATCH_SIZE
    for batch_start in range(0, len(news_items), BATCH_SIZE):
        batch = news_items[batch_start:batch_start + BATCH_SIZE]

        # Build numbered list for this batch
        items_list = ""
        for i, item in enumerate(batch):
            items_list += f"{i+1}. {item['title']} — {item['snippet'][:150]}\n"

        prompt = BATCH_CLASSIFICATION_PROMPT.format(items_list=items_list)

        try:
            result = chain.invoke(prompt).strip()

            # Parse JSON array
            cleaned = result
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            classifications = json.loads(cleaned)

            # Apply classifications to each item in this batch
            for i, item in enumerate(batch):
                if i < len(classifications):
                    class_num = int(classifications[i])
                    if class_num < 1 or class_num > 6:
                        class_num = 6
                else:
                    class_num = 6

                item["news_class"] = class_num
                item["classification_confidence"] = 1.0
                item["status"] = "classified"

            print(f"[Classifier] Batch {batch_start//BATCH_SIZE + 1}: classified {len(batch)} items")

        except Exception as e:
            print(f"[Classifier] Batch {batch_start//BATCH_SIZE + 1} failed: {e}")
            for item in batch:
                item["news_class"] = 6
                item["classification_confidence"] = 0.0
                item["status"] = "classified"

    state["items"] = news_items
    print(f"[Classifier] Done. Classified {len(news_items)} total items.")
    return state
