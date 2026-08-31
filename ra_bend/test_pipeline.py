"""
Test: Full pipeline — Scout → Classifier → Sourcer → Writer
Fetches 32, passes 8 through the full pipeline.
"""
import sys
sys.path.insert(0, ".")

from agents.scout import scout_global_state_update
from agents.classifier import classify_news
from agents.sourcer import make_distilled_context
from agents.writer import news_writer

# Step 1: Scout
print("=" * 70)
print("STEP 1: SCOUT")
print("=" * 70)
state = scout_global_state_update()
print(f"Fetched {len(state['items'])} items")
state["items"] = state["items"][:8]
print(f"Passing 8 items to pipeline\n")

# Step 2: Classifier
print("=" * 70)
print("STEP 2: CLASSIFIER")
print("=" * 70)
state = classify_news(state)
print()

# Step 3: Sourcer
print("=" * 70)
print("STEP 3: SOURCER")
print("=" * 70)
state = make_distilled_context(state)
print()

# Step 4: Writer
print("=" * 70)
print("STEP 4: WRITER")
print("=" * 70)
state = news_writer(state)
print()

# ─── Print Results ───
print("=" * 70)
print("FINAL OUTPUT")
print("=" * 70)
for i, item in enumerate(state["items"]):
    print(f"\n{'─'*70}")
    print(f"ITEM {i+1} | Class: {item['news_class']} | Status: {item['status']}")
    print(f"Title: {item['title'][:60]}")
    print(f"{'─'*70}")

    draft = item.get("writer_draft")
    if draft:
        print(f"\n--- DRAFT ---")
        print(draft["full_content"][:500])
        print(f"\n--- 6-POINT SUMMARY ---")
        print(draft["summary"])
    else:
        print("[No draft generated]")
        if item.get("error_log"):
            print(f"Errors: {item['error_log']}")
