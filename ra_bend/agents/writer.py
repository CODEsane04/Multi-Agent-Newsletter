# Writer Agent — Sonnet for drafting, Nova-Pro for summarization

import os
from openai import OpenAI
from graph.state import PipelineState
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

# ─── Prompts ───

GENERAL_PROMPT = """You are a senior AI journalist writing for a weekly internal newsletter read by software engineers and tech leads.

Write a concise, technically informative newsletter section about the following news item.

STRUCTURE (follow this exactly):
1. **Headline** — A sharp, informative title (max 10 words)
2. **TL;DR** — One sentence summary (max 25 words)
3. **What happened** — 2-3 sentences explaining the core news
4. **Why it matters** — 2-3 sentences on engineering/business impact
5. **Key details** — 3-5 bullet points with specific facts, numbers, or quotes from the article

RULES:
- Total length: 150-250 words
- No marketing fluff ("revolutionary", "game-changing")
- No speculation — only state what the article confirms
- Use plain, direct language
- If technical details are available, include them
- End with the source URL for further reading

NEWS CLASS: {news_class}
TITLE: {title}
SNIPPET: {snippet}

FULL ARTICLE:
{full_article}

Write the newsletter section now:"""


RESEARCH_PROMPT = """You are a senior AI research journalist writing for a weekly internal newsletter read by ML engineers and research-oriented tech leads.

Write a technically rigorous newsletter section about the following research paper/finding.

STRUCTURE (follow this exactly):
1. **Headline** — A sharp, informative title (max 10 words)
2. **TL;DR** — One sentence summary (max 25 words)
3. **The paper** — 2-3 sentences: Who published it, what problem it addresses, and the core contribution
4. **How it works** — 3-4 sentences explaining the technical approach or novel mechanism. Do not oversimplify. Use precise terminology.
5. **Why it matters** — 2-3 sentences on implications for practitioners (what changes in how we build/train/deploy)
6. **Key details** — 3-5 bullet points with specific metrics, datasets, architecture choices, or limitations mentioned

RULES:
- Total length: 200-300 words
- No marketing fluff — this is for engineers who read papers
- No speculation — only state what the article/abstract confirms
- Use precise technical language (name the architecture, method, loss function if mentioned)
- If the paper abstract is provided, ground your explanation in it
- End with the source URL for further reading

TITLE: {title}
SNIPPET: {snippet}

FULL ARTICLE:
{full_article}

PAPER AUTHORS: {paper_author}
PAPER ABSTRACT: {paper_abstract}

Write the newsletter section now:"""


SUMMARY_PROMPT = """You are given a newsletter draft. Extract exactly 6 key factual claims from it.

Each claim must be a single, specific, verifiable statement (a fact, number, name, or action that can be checked against a source).

RULES:
- Output ONLY a numbered list of 6 claims
- Each claim must be one sentence
- Focus on quantitative facts, named entities, specific actions, and technical claims
- Do NOT include opinions or analysis — only checkable facts

DRAFT:
{draft}

List the 6 factual claims:"""


REWRITE_PROMPT = """You are a senior AI journalist rewriting a newsletter section that was rejected by a fact-checker.

The previous draft was rejected for the following reason:
REVIEWER FEEDBACK: {reviewer_note}

Your task: Rewrite the newsletter section, fixing the issues raised by the reviewer. Use ONLY facts confirmed in the source material below.

STRUCTURE (follow this exactly):
1. **Headline** — A sharp, informative title (max 10 words)
2. **TL;DR** — One sentence summary (max 25 words)
3. **What happened** — 2-3 sentences explaining the core news
4. **Why it matters** — 2-3 sentences on engineering/business impact
5. **Key details** — 3-5 bullet points with specific facts, numbers, or quotes from the article

RULES:
- Total length: 150-250 words
- Fix ALL issues mentioned in the reviewer feedback
- No speculation — only state what the source confirms
- If a claim cannot be verified from the source, remove it entirely

NEWS CLASS: {news_class}
TITLE: {title}

SOURCE MATERIAL:
{full_article}

{arxiv_section}

Rewrite the newsletter section now:"""


# ─── Schema ───

class writer_draft_schema(TypedDict):
    full_content: str
    summary: list[str]


# ─── Writer Function ───

def news_writer(state: PipelineState) -> PipelineState:
    news_items = state["items"]

    client = OpenAI(
        base_url=os.getenv("BEDROCK_BASE_URL"),
        api_key=os.getenv("BEDROCK_API_KEY"),
        timeout=120.0,
    )

    class_labels = {
        1: "Model & Foundation Release",
        2: "Academic Research & Paper",
        3: "Product & Application Launch",
        4: "Hardware & Infrastructure",
        5: "Business, Policy & Geopolitics",
        6: "General / Miscellaneous",
    }

    for item in news_items:
        dc = item.get("distilled_context", {})
        item_class = item["news_class"]
        reviewer_status = item.get("reviewer_status")

        # ─── Skip if already approved ───
        if reviewer_status == "APPROVED":
            continue

        # ─── Build the prompt based on status ───
        if reviewer_status == "REJECTED":

            #update the number of iterations
            iterations = state["iterations"]
            iterations = iterations + 1
            state["iterations"] = iterations
            
            # Rewrite mode — use reviewer feedback + source material
            arxiv_section = ""
            if item_class == 2:
                arxiv = dc.get("arxiv_data") or {}
                if arxiv.get("paper_author"):
                    arxiv_section += f"PAPER AUTHORS: {arxiv['paper_author']}\n"
                if arxiv.get("paper_abstract"):
                    arxiv_section += f"PAPER ABSTRACT: {arxiv['paper_abstract']}\n"

            prompt = REWRITE_PROMPT.format(
                reviewer_note=item.get("reviewer_reasoning", "No specific feedback."),
                news_class=class_labels.get(item_class, "General"),
                title=dc.get("title", item["title"]),
                full_article=dc.get("full_article", "")[:2000],
                arxiv_section=arxiv_section,
            )
            print(f"[Writer] Rewriting (attempt {item['retry_count'] + 1}): {item['title'][:50]}")

        else:
            # First-time generation (reviewer_status is None)
            if item_class == 2:
                arxiv = dc.get("arxiv_data") or {}
                prompt = RESEARCH_PROMPT.format(
                    title=dc.get("title", item["title"]),
                    snippet=dc.get("snippet", item["snippet"]),
                    full_article=dc.get("full_article", "")[:2000],
                    paper_author=arxiv.get("paper_author", "Not available"),
                    paper_abstract=arxiv.get("paper_abstract", "Not available"),
                )
            else:
                prompt = GENERAL_PROMPT.format(
                    news_class=class_labels.get(item_class, "General"),
                    title=dc.get("title", item["title"]),
                    snippet=dc.get("snippet", item["snippet"]),
                    full_article=dc.get("full_article", "")[:2000],
                )

        # ─── Phase 1: Generate/rewrite draft with Sonnet ───
        try:
            resp = client.chat.completions.create(
                model="sonnet",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3,
            )
            draft_content = resp.choices[0].message.content

            # ─── Phase 2: Summarize into 6 claims with Nova-Pro ───
            summary_prompt = SUMMARY_PROMPT.format(draft=draft_content)

            resp2 = client.chat.completions.create(
                model="nova-pro",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=300,
                temperature=0.1,
            )
            summary_text = resp2.choices[0].message.content

            # Store in state
            item["writer_draft"] = {
                "full_content": draft_content,
                "summary": summary_text,
            }
            item["status"] = "drafted"
            item["retry_count"] += 1
            print(f"[Writer] Drafted: {item['title'][:50]}")

        except Exception as e:
            print(f"[Writer] Error for '{item['title'][:50]}': {e}")
            item["writer_draft"] = None
            item["status"] = "drafted"
            item["error_log"].append(f"[Writer] {e}")

    state["items"] = news_items
    print(f"[Writer] Done. Drafted {len(news_items)} items.")
    return state
