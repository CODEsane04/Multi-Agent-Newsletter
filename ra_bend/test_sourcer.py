"""
Test: Scout → Classifier → Sourcer pipeline
Uses 16 items (2 classifier batches), then prints distilled context of first 3.
"""
import sys
sys.path.insert(0, ".")

from agents.scout import scout_global_state_update
from agents.classifier import classify_news
from agents.sourcer import make_distilled_context

# Step 1: Fetch
state = scout_global_state_update()
state["items"] = state["items"][:16]
print(f"Fetched {len(state['items'])} items\n")

# Step 2: Classify
state = classify_news(state)
print()

# Step 3: Enrich with full article
state = make_distilled_context(state)
print()

# Print distilled context of first 3 items
print(f"{'='*70}")
print("DISTILLED CONTEXT (first 3 items)")
print(f"{'='*70}")
for item in state["items"][:3]:
    print(f"\nTitle: {item['title'][:60]}")
    print(f"Class: {item['news_class']}")
    dc = item.get("distilled_context")
    if dc:
        text = dc.get("full_article", "")
        print(f"Full article length: {len(text.split())} words")
        print(f"Preview: {text[:200]}...")
    else:
        print("No distilled context")
    print("-" * 50)
