# SLM Classifier Router — Batch classification via Gemma (batches of 8)

from langchain_google_genai import ChatGoogleGenerativeAI
from graph.state import PipelineState
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
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

BATCH_SIZE = 8


class batch_op_schema(BaseModel):
    classifications: list[int] = Field(
        description="List of class numbers (1 to 6) corresponding to each news item in order."
    )
    confidence: list[int] = Field(
        description="out of 10, how confident you are about a news_item belonging to your predicted class"
    )


llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0.1)

structured_llm = llm.with_structured_output(
    schema=batch_op_schema.model_json_schema(), method="json_schema"
)

BATCH_CLASSIFICATION_PROMPT = PromptTemplate(
    template="""You are a news classifier. Given list of news items, Classify each news item below into exactly one class:

Classes:
1. Release of a new frontier model or an improved model
2. New Academic research or paper
3. New AI product or Application launch
4. Hardware or Infrastructure
5. Business policy and Geopolitics
6. General or miscellaneous

News items:
{items_list}

Respond with ONLY JSON arrays:

classifications: where each element is the class number (1-6) for the corresponding item, in order.
Example response for 3 items: [3, 5, 1]

confidence: where each element is the confidence score (out of 10) for the corresponding prediction of class of a news item, in order. Example response for 3 items: [9, 5, 7]

Return ONLY the JSON array. No explanation.""",
    input_variables=["items_list"],
)
chain = BATCH_CLASSIFICATION_PROMPT | structured_llm


def classify_news(state: PipelineState) -> PipelineState:
    news_items = state["items"]
    if not news_items:
        return state

    # Process in batches of BATCH_SIZE
    for batch_start in range(0, len(news_items), BATCH_SIZE):
        batch = news_items[batch_start : batch_start + BATCH_SIZE]

        # Build numbered list for this batch
        items_list = ""
        for i, item in enumerate(batch):
            items_list += f"{i+1}. {item['title']} | {item['snippet'][:400]}\n"

        try:
            response = chain.invoke({"items_list": items_list})

            raw_class_list = response["classifications"]
            classification_confidence_list = response["confidence"]

            # Apply classifications to each item in this batch
            for i, item in enumerate(batch):
                if i < len(raw_class_list):
                    class_num = int(raw_class_list[i])
                    if class_num < 1 or class_num > 6:
                        class_num = 6
                else:
                    class_num = 6

                if i < len(classification_confidence_list):
                    confi = float(classification_confidence_list[i] / 10)
                else:
                    confi = 1.0

                item["news_class"] = class_num
                item["classification_confidence"] = confi
                item["status"] = "classified"

                print(f"  [{i+1}] {item['title'][:60]} -> Class: {class_num} | Conf: {confi}")

            print(f"[Classifier] Batch {batch_start//BATCH_SIZE + 1}: classified {len(batch)} items\n")

        except Exception as e:
            print(f"[Classifier] Batch {batch_start//BATCH_SIZE + 1} failed: {e}")
            for item in batch:
                item["news_class"] = 6
                item["classification_confidence"] = 0.0
                item["status"] = "classified"

    state["items"] = news_items
    print(f"[Classifier] Done. Classified {len(news_items)} total items.")
    return state
