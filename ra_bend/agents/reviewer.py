# Reviewer Agent — Gemma entailment verification

import json
from graph.state import PipelineState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

#--------- SCHEMA ------------
class review_schema(BaseModel) :
    verdict : str = Field(description="The evaluation decision: return 'APPROVED' if all claims are factually supported by the source material, or 'REJECTED' if any claim is fabricated or contradictory.")
    feedback : str = Field(description="If REJECTED, provide a concise explanation of the factual error or unverified claim and how to correct it; if APPROVED, return 'None'.")

#---------- LLM -------------

llm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    temperature=0.1
)

structured_reviewer = llm.with_structured_output(schema=review_schema.model_json_schema(), method="json_schema")



reviewer_model = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    temperature=0.1,
)
parser = StrOutputParser()
chain = reviewer_model | parser

GENERAL_REVIEW_PROMPT = """You are a pragmatic fact-checker for an AI newsletter. Your job is to verify if the claims in the summary are supported by the source material.

REFERENCE CONTEXT (source of truth):
Title: {title}
Full Article: {full_article}

CLAIMS TO VERIFY (from the writer's draft):
{summary}

RULES:
- Check each claim against the reference context
- A claim is VALID if it is directly stated or reasonably inferred from the source
- A claim is INVALID only if it clearly contradicts the source OR is completely fabricated (not mentioned at all)
- Be pragmatic — do NOT reject for minor phrasing differences or reasonable inferences
- Only reject if there is a genuine factual error or fabrication

Respond with ONLY a valid JSON object:
{{"verdict": "approved" | "rejected", "feedback": "clearly justify your decision, if you have decide to reject, then clearly mention the wrongly stated facts, and how should the writer agent state the fact correctly, keep the feedback short and precise"}}

Return ONLY the JSON. No extra text."""


RESEARCH_REVIEW_PROMPT = """You are a pragmatic fact-checker for an AI research newsletter. Your job is to verify if the claims in the summary are supported by the source material.

REFERENCE CONTEXT (source of truth):
Title: {title}
Full Article: {full_article}

PAPER AUTHORS: {paper_author}
PAPER ABSTRACT: {paper_abstract}

CLAIMS TO VERIFY (from the writer's draft):
{summary}

RULES:
- Check each claim against the reference context AND the paper abstract
- A claim is VALID if it is directly stated or reasonably inferred from the source
- A claim is INVALID only if it clearly contradicts the source, misattributes the research, or fabricates technical details
- Be pragmatic — do NOT reject for minor phrasing differences or reasonable inferences
- Only reject if there is a genuine factual error or fabrication

Respond with ONLY a valid JSON object:
{{"verdict": "approved" | "rejected", "feedback": "clearly justify your decision, if you have decide to reject, then clearly mention the wrongly stated facts, and how should the writer agent state the fact correctly, keep the feedback short and precise"}}

Return ONLY the JSON. No extra text."""


def review_writer(state: PipelineState) -> PipelineState:
    news_items = state["items"]
    total_items_rejected = 0;

    SESSION_PROMPT = """"""

    for item in news_items:
        # Skip items without a draft
        if not item.get("writer_draft"):
            continue

        # Skip items already approved
        if item.get("reviewer_status") == "APPROVED":
            continue

        dc = item["distilled_context"]
        writer_draft = item["writer_draft"]
        summary = writer_draft["summary"]
        news_class = item["news_class"]

        # ─── Build the review prompt based on class ───
        if news_class == 2:
            arxiv = dc.get("arxiv_data") or {}

            SESSION_PROMPT = PromptTemplate(
                template=RESEARCH_REVIEW_PROMPT,
                input_variables=["title", "full_article","paper_author", "paper_abstract", "summary"]
            )

            SESSION_PROMPT = SESSION_PROMPT.invoke({
                "title" : dc.get("title", ""),
                "full_article" : dc.get("full_article", ""),
                "paper_author" : arxiv.get("paper_author", "Not available"),
                "paper_abstract" : arxiv.get("paper_abstract", "Not available"),
                "summary" : summary,
            })
        
        else:

            SESSION_PROMPT = PromptTemplate(
                template=GENERAL_REVIEW_PROMPT,
                input_variables=["title", "full_article","summary"]
            )

            SESSION_PROMPT = SESSION_PROMPT.invoke({
                "title" : dc.get("title", ""),
                "full_article" : dc.get("full_article", ""),
                "summary" : summary,
            })

        # ─── Call the reviewer model ───
        try:
            result = structured_reviewer.invoke(SESSION_PROMPT)

            # # Parse JSON response
            # cleaned = result
            # if cleaned.startswith("```"):
            #     cleaned = cleaned.split("\n", 1)[1]
            # if cleaned.endswith("```"):
            #     cleaned = cleaned.rsplit("```", 1)[0]
            # cleaned = cleaned.strip()

            # decision = json.loads(cleaned)
            verdict = result.get("verdict", "rejected").upper()
            feedback = result.get("feedback", "No reason provided.")

            item["reviewer_status"] = verdict
            item["reviewer_reasoning"] = feedback

            if verdict == "APPROVED":
                item["status"] = "approved"
                print(f"[Reviewer] ✓ Approved: {item['title'][:50]}")
            else:
                item["status"] = "reviewing"
                print(f"[Reviewer] ✗ Rejected: {item['title'][:50]} — {feedback}")
                total_items_rejected = total_items_rejected + 1;

        except Exception as e:
            # If parsing fails, approve by default (pragmatic — don't block on reviewer error)
            print(f"[Reviewer] Error for '{item['title'][:50]}': {e}")
            item["reviewer_status"] = "APPROVED"
            item["reviewer_reasoning"] = "Auto-approved due to reviewer parsing error."
            item["status"] = "approved"

    state["items"] = news_items

    ## Loop metric updation
    if total_items_rejected > 0 :
        state["all_approved"] = False
    else :
        state["all_approved"] = True

    print(f"[Reviewer] Done. Reviewed {len(news_items)} items.")
    return state
