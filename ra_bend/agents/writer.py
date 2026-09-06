# Writer Agent — Gemma-4-31b-it for drafting, Gemma-4-26b-a4b-it for summarization

import os
from openai import OpenAI
from graph.state import PipelineState
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ───----------- Prompts ───-------------------

# GENERAL_PROMPT = """You are a senior AI journalist writing for a weekly internal newsletter read by software engineers and tech leads.

# Write a concise, technically informative newsletter section about the following news item.

# STRUCTURE (follow this exactly):
# 1. **Headline** — A sharp, informative title (max 10 words)
# 2. **TL;DR** — One sentence summary (max 25 words)
# 3. **What happened** — 2-3 sentences explaining the core news
# 4. **Why it matters** — 2-3 sentences on engineering/business impact
# 5. **Key details** — 3-5 bullet points with specific facts, numbers, or quotes from the article

# RULES:
# - Total length: 150-250 words
# - No marketing fluff ("revolutionary", "game-changing")
# - No speculation — only state what the article confirms
# - Use plain, direct language
# - If technical details are available, include them
# - End with the source URL for further reading

# NEWS CLASS: {news_class}
# TITLE: {title}
# SNIPPET: {snippet}

# FULL ARTICLE:
# {full_article}

# Write the newsletter section now:"""


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

the class specific flow : {class_prompt},

TITLE: {title}

SOURCE MATERIAL:
{full_article}

if it is a research news : research paper data : author : {author}, abstract : {abstract}

Rewrite the newsletter section now:"""

GEN_RULES = """GENERAL EDITORIAL RULES:
- Write factual, neutral, publication-ready AI news.
- Do not invent missing information.
- Attribute company/researcher/government claims.
- Distinguish facts, claims, evidence and predictions.
- Avoid promotional or sensational language.
- Use specific names, dates, numbers and comparisons where available.
- Avoid repeating information.
- The main_content must be detailed and descriptive, strictly following the provided class-specific schema.
- The article must strictly follow this output schema:
  header : str
  introduction : str
  main_content : str
  outro : str
- End with the most relevant next step, uncertainty or open question."""

PROMPT_LIST = ["""filler""", """CLASS 1 — MODEL RELEASE

HEADLINE:
Company + model + key new capability/improvement. Factual, no hype.

INTRO:
State who released what, when, the main improvement, and why it matters.

MAIN:
1. What was released — model/version, capabilities, availability.
2. What changed — improvements vs previous version.
3. Performance — benchmarks, comparisons, and whether evidence is company-reported or independent.
4. Access & cost — pricing, API/product availability, rollout.
5. Competitive context — relevant competing models.
6. Caveats — limitations, benchmark gaps, reliability/safety concerns.

OUTRO:
State rollout/next steps and what remains to be established.""", """CLASS 2 — RESEARCH

HEADLINE:
Lead with the key research finding, not the paper's technical title.

INTRO:
State who conducted the research, the central finding, where/when it was published, and its significance.

MAIN:
1. Finding — explain the discovery in plain language.
2. Method — models, data, experiment and evaluation approach.
3. Results — strongest evidence and quantitative findings.
4. Interpretation — what researchers say the results mean.
5. Limitations — methodology, data, sample, generalization or replication issues.
6. Independent context — outside validation, criticism or supporting evidence if available.

OUTRO:
State what remains uncertain and what future testing/replication is needed.""", """CLASS 3 — PRODUCT LAUNCH

HEADLINE:
Product + concrete user-facing capability. Avoid marketing language.

INTRO:
State what launched, who launched it, who it is for, and what users can actually do with it.

MAIN:
1. Product — what it is.
2. Use cases — concrete tasks/workflows users can perform.
3. How it works — model, agents, RAG, tools or integrations only where relevant.
4. Access & pricing — users, plans, API, regions, rollout.
5. Privacy/security — data handling and permissions where relevant.
6. Competition — what existing product/workflow it competes with.
7. Limitations — reliability, restrictions or missing capabilities.

OUTRO:
State rollout status and what real-world adoption/use will reveal.""", """CLASS 4 — HARDWARE / INFRASTRUCTURE

HEADLINE:
Hardware/infrastructure + main performance, efficiency or scaling benefit.

INTRO:
State what was introduced, by whom, its purpose, and why it matters for AI infrastructure.

MAIN:
1. Announcement — chip/system/infrastructure and generation.
2. Technical specs — only relevant metrics.
3. Performance — comparison with previous generation/competitors.
4. Economics — cost, performance-per-watt, inference/training implications.
5. Supply & deployment — production, availability, cloud access, timeline.
6. AI impact — implications for training, inference, data centers or scaling.

OUTRO:
State deployment timeline, availability and the key factor determining adoption.""", """CLASS 5 — POLICY / GEOPOLITICS

HEADLINE:
State the policy/business action and immediate affected party or consequence. Neutral, factual.

INTRO:
State who acted, what changed, when, and the immediate impact.

MAIN:
1. What happened — precise policy, decision, restriction, agreement, lawsuit, etc.
2. Why — stated rationale from the relevant actor(s), clearly attributed.
3. Who is affected — companies, governments, chipmakers, AI firms, consumers, etc.
4. Positions — relevant perspectives from affected or opposing stakeholders.
5. Impact — separate confirmed effects from forecasts/predictions.
6. Background — only context necessary to understand the event.
7. Next steps — deadlines, implementation, negotiations, court action, etc.

OUTRO:
End with the next concrete development or unresolved policy consequence.""", """CLASS 6 — GENERAL

HEADLINE:
Summarize the central event accurately and concisely.

INTRO:
Answer WHO + WHAT + WHEN + WHY IT MATTERS.

MAIN:
1. What happened — confirmed facts.
2. Why it matters — practical/industry significance.
3. Evidence — numbers, statements, documents or other supporting details.
4. Context — necessary background.
5. Reactions — relevant attributed responses.
6. Unknowns — unresolved or unverified information.

OUTRO:
State the next meaningful development, pending decision or open question."""]

prompt_rest = PromptTemplate(
    template="""given the general rules of writing draft : {GEN_RULES},

        the class specific flow : {class_prompt},

        the actual news : title : {title}, full article : {full_article}
    """,
    input_variables=["GEN_RULES", "class_prompt", "title", "full_article"]
)

prompt_research = PromptTemplate(
    template="""given the general rules of writing draft : {GEN_RULES},

        And following are the specific details : 

        the class specific flow : {class_prompt},

        the actual news : title : {title}, full article : {full_article},

        arxhiv research paper data, paper author : {author}, paper_abstract : {abstract},
    """,
    input_variables=["GEN_RULES", "class_prompt", "author", "abstract", "title", "full_article"]
)

prompt_sum = PromptTemplate(
    template=SUMMARY_PROMPT,
    input_variables=["draft"]
)

prompt_rewrite = PromptTemplate(
    template=REWRITE_PROMPT,
    input_variables=["reviewer_note", "class_prompt", "title", "full_article", "author", "abstract"]
)
# ───-------------- Schema ───----------------

class writer_draft_schema(TypedDict):
    full_content: str
    summary: list[str]

class draft_schema(BaseModel):

    header: str = Field(
        description="Write a concise, factual headline that clearly states the main news and its significance."
    )

    introduction: str = Field(
        description="Briefly summarize what happened, who is involved, when it happened, and why it matters."
    )

    main_content: str = Field(
        description="Provide a detailed, descriptive account of the news following the class-specific editorial structure, evidence, context, and key details."
    )

    outro: str = Field(
        description="Conclude with the most relevant next step, unresolved question, future implication, or remaining uncertainty."
    )

class summary_schema(BaseModel) :
    summary : str = Field(description="A numbered list of 6 specific, verifiable factual claims extracted directly from the draft for fact-checking.")

# ───--------- LLMs ───--------------

llm1 = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    temperature=0.7
)

llm2 = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    temperature=0.1
)

structured_writer = llm1.with_structured_output(schema=draft_schema.model_json_schema(), method="json_schema")

sumarry_writer = llm2.with_structured_output(schema=summary_schema.model_json_schema(), method="json_schema")


# ───------------- Writer Function ───----------------

def news_writer(state: PipelineState) -> PipelineState:
    news_items = state["items"]

    class_labels = {
        1: "Model & Foundation Release",
        2: "Academic Research & Paper",
        3: "Product & Application Launch",
        4: "Hardware & Infrastructure",
        5: "Business, Policy & Geopolitics",
        6: "General / Miscellaneous",
    }

    SESSION_PROMPT = """"""

    for item in news_items:
        dc = item.get("distilled_context", {})
        item_class = item["news_class"]
        reviewer_status = item.get("reviewer_status")

        title = ""
        full_article = ""

        title = dc.get("title", item.get("title", ""))
        full_article = dc.get("full_article", "")


        # ─── Skip if already approved ───
        if reviewer_status == "APPROVED":
            continue
        if reviewer_status == "REJECTED":
        
            #update the number of iterations
            iterations = state["iterations"]
            iterations = iterations + 1
            state["iterations"] = iterations

            reviewer_feedback = item.get("reviewer_reasoning") or "No specific feedback, only mention facts present in the full article"

            author = ""
            abstract = ""
            
            # Rewrite mode — use reviewer feedback + source material
            
            if item_class == 2:
                arxiv_data = dc.get("arxiv_data") or {}
                
                if arxiv_data.get("paper_author") :
                    author = arxiv_data.get("paper_author")
                if arxiv_data.get("paper_abstract") :
                    abstract = arxiv_data.get("paper_abstract")

            SESSION_PROMPT = prompt_rewrite.invoke({
                "reviewer_note" : reviewer_feedback,
                "class_prompt" : PROMPT_LIST[item_class],
                "title" : title,
                "full_article" : full_article,
                "author" : author,
                "abstract" : abstract,
            })
            
            print(f"[Writer] Rewriting (attempt {item['retry_count'] + 1}): {item['title'][:50]}")
        else :

            if item_class == 2 :
                arxiv_data = dc.get("arxiv_data") or {}
                author = ""
                abstract = ""

                if arxiv_data.get("paper_author") :
                    author = arxiv_data.get("paper_author")
                if arxiv_data.get("paper_abstract") :
                    abstract = arxiv_data.get("paper_abstract")

                SESSION_PROMPT = prompt_research.invoke({
                    "GEN_RULES" : GEN_RULES,
                    "class_prompt" : PROMPT_LIST[item_class],
                    "author" : author,
                    "abstract" : abstract,
                    "title" : title,
                    "full_article" : full_article
                    
                })
            else :
                SESSION_PROMPT = prompt_rest.invoke({
                    "GEN_RULES" : GEN_RULES,
                    "class_prompt" : PROMPT_LIST[item_class],
                    "title" : title,
                    "full_article" : full_article
                })

        try :

            draft_content = structured_writer.invoke(SESSION_PROMPT)
            print("\n Done drafting news, sending for summarization \n")

            #---------get the summary-----------

            summary_chain = prompt_sum | sumarry_writer
            summary_content = summary_chain.invoke({
                "draft" : draft_content
            })

            print("\n Done summarizing news, updating the new_item's state \n")

            # Store in state

            item["writer_draft"] = {
                "writer_system_prompt" : PROMPT_LIST[item_class],
                "full_content": f" header : {draft_content['header']}, intro : {draft_content['introduction']}, main_content : {draft_content['main_content']}, outro : {draft_content['outro']}",
                "summary": summary_content["summary"],
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

    # for item in news_items:
    #     dc = item.get("distilled_context", {})
    #     item_class = item["news_class"]
    #     reviewer_status = item.get("reviewer_status")

    #     # ─── Skip if already approved ───
    #     if reviewer_status == "APPROVED":
    #         continue

    #     # ─── Build the prompt based on status ───
    #     if reviewer_status == "REJECTED":

    #         #update the number of iterations
    #         iterations = state["iterations"]
    #         iterations = iterations + 1
    #         state["iterations"] = iterations
            
    #         # Rewrite mode — use reviewer feedback + source material
    #         arxiv_section = ""
    #         if item_class == 2:
    #             arxiv = dc.get("arxiv_data") or {}
    #             if arxiv.get("paper_author"):
    #                 arxiv_section += f"PAPER AUTHORS: {arxiv['paper_author']}\n"
    #             if arxiv.get("paper_abstract"):
    #                 arxiv_section += f"PAPER ABSTRACT: {arxiv['paper_abstract']}\n"

    #         prompt = REWRITE_PROMPT.format(
    #             reviewer_note=item.get("reviewer_reasoning", "No specific feedback."),
    #             news_class=class_labels.get(item_class, "General"),
    #             title=dc.get("title", item["title"]),
    #             full_article=dc.get("full_article", "")[:2000],
    #             arxiv_section=arxiv_section,
    #         )
    #         print(f"[Writer] Rewriting (attempt {item['retry_count'] + 1}): {item['title'][:50]}")

    #     else:
    #         # First-time generation (reviewer_status is None)
    #         if item_class == 2:
    #             arxiv = dc.get("arxiv_data") or {}
    #             prompt = RESEARCH_PROMPT.format(
    #                 title=dc.get("title", item["title"]),
    #                 snippet=dc.get("snippet", item["snippet"]),
    #                 full_article=dc.get("full_article", "")[:2000],
    #                 paper_author=arxiv.get("paper_author", "Not available"),
    #                 paper_abstract=arxiv.get("paper_abstract", "Not available"),
    #             )
    #         else:
    #             prompt = GENERAL_PROMPT.format(
    #                 news_class=class_labels.get(item_class, "General"),
    #                 title=dc.get("title", item["title"]),
    #                 snippet=dc.get("snippet", item["snippet"]),
    #                 full_article=dc.get("full_article", "")[:2000],
    #             )

    #     # ─── Phase 1: Generate/rewrite draft with Sonnet ───
    #     try:
    #         resp = client.chat.completions.create(
    #             model="sonnet",
    #             messages=[{"role": "user", "content": prompt}],
    #             max_tokens=600,
    #             temperature=0.3,
    #         )
    #         draft_content = resp.choices[0].message.content

    #         # ─── Phase 2: Summarize into 6 claims with Nova-Pro ───
    #         summary_prompt = SUMMARY_PROMPT.format(draft=draft_content)

    #         resp2 = client.chat.completions.create(
    #             model="nova-pro",
    #             messages=[{"role": "user", "content": summary_prompt}],
    #             max_tokens=300,
    #             temperature=0.1,
    #         )
    #         summary_text = resp2.choices[0].message.content

    #         # Store in state
    #         item["writer_draft"] = {
    #             "full_content": draft_content,
    #             "summary": summary_text,
    #         }
    #         item["status"] = "drafted"
    #         item["retry_count"] += 1
    #         print(f"[Writer] Drafted: {item['title'][:50]}")

    #     except Exception as e:
    #         print(f"[Writer] Error for '{item['title'][:50]}': {e}")
    #         item["writer_draft"] = None
    #         item["status"] = "drafted"
    #         item["error_log"].append(f"[Writer] {e}")

    # state["items"] = news_items
    # print(f"[Writer] Done. Drafted {len(news_items)} items.")
    # return state
