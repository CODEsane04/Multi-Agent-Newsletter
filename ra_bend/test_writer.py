"""
Test: Full pipeline (Scout → Classifier → Sourcer) then Writer on 2 items.
Uses Bedrock Sonnet (Claude 4.6) for writing.
"""
import sys
sys.path.insert(0, ".")

from agents.scout import scout_global_state_update
from agents.classifier import classify_news
from agents.sourcer import make_distilled_context
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── Pipeline: Scout → Classify → Enrich (5 items only) ───
print("Step 1: Fetching news items...")
state = scout_global_state_update()
state["items"] = state["items"][:5]
print(f"Using {len(state['items'])} items\n")

print("Step 2: Classifying...")
state = classify_news(state)
print()

print("Step 3: Building distilled context...")
state = make_distilled_context(state)
print()

# ─── Writer: Take 2 items and generate newsletter sections ───
print("Step 4: Writing newsletter sections for 2 items...\n")

client = OpenAI(
    base_url=os.getenv("BEDROCK_BASE_URL"),
    api_key=os.getenv("BEDROCK_API_KEY"),
    timeout=120.0,
)

WRITER_PROMPT = """You are a senior AI journalist writing for a weekly internal newsletter read by software engineers and tech leads.

Your task: Write a concise, technically informative newsletter section about the following news item.

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
- End with the source link for further reading

NEWS CLASS: {news_class}
TITLE: {title}
SNIPPET: {snippet}

FULL ARTICLE:
{full_article}

{arxiv_section}

Write the newsletter section now:"""


def write_news_section(item: dict) -> str:
    dc = item.get("distilled_context", {})

    # Build arxiv section if available
    arxiv_section = ""
    if dc.get("arxiv_data"):
        arxiv = dc["arxiv_data"]
        if arxiv.get("paper_author"):
            arxiv_section += f"PAPER AUTHORS: {arxiv['paper_author']}\n"
        if arxiv.get("paper_abstract"):
            arxiv_section += f"PAPER ABSTRACT: {arxiv['paper_abstract']}\n"

    # Map class number to label for the prompt
    class_labels = {
        1: "Model & Foundation Release",
        2: "Academic Research & Paper",
        3: "Product & Application Launch",
        4: "Hardware & Infrastructure",
        5: "Business, Policy & Geopolitics",
        6: "General / Miscellaneous",
    }

    prompt = WRITER_PROMPT.format(
        news_class=class_labels.get(item.get("news_class", 6), "General"),
        title=dc.get("title", item["title"]),
        snippet=dc.get("snippet", item["snippet"]),
        full_article=dc.get("full_article", "")[:2000],
        arxiv_section=arxiv_section,
    )

    resp = client.chat.completions.create(
        model="sonnet",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# Write for first 2 items
for i, item in enumerate(state["items"][:2]):
    print(f"{'='*70}")
    print(f"ITEM {i+1}: {item['title'][:60]}")
    print(f"CLASS: {item['news_class']}")
    print(f"{'='*70}\n")

    try:
        output = write_news_section(item)
        print(output)
    except Exception as e:
        print(f"[Writer] Error: {e}")

    print(f"\n{'─'*70}\n")
