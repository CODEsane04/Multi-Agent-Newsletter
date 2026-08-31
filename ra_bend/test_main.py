"""
Production Test: Full pipeline with iterative Writer ↔ Reviewer loop.
Fetches 32, processes 16 through all agents.
"""
import sys
sys.path.insert(0, ".")

from agents.scout import scout_global_state_update
from agents.dedup import remove_duplicates
from agents.classifier import classify_news
from agents.sourcer import make_distilled_context
from agents.writer import news_writer
from agents.reviewer import review_writer

MAX_REVIEW_CYCLES = 3

# ─── Step 1: Scout ───
print("=" * 70)
print("STEP 1: SCOUT")
print("=" * 70)
state = scout_global_state_update()
print(f"Fetched {len(state['items'])} items")
state["items"] = state["items"][:16]
print(f"Processing 16 items\n")

# ─── Step 2: Dedup ───
print("=" * 70)
print("STEP 2: DEDUPLICATION")
print("=" * 70)
state = remove_duplicates(state)
print()

# ─── Step 3: Classifier ───
print("=" * 70)
print("STEP 3: CLASSIFIER")
print("=" * 70)
state = classify_news(state)
print()

# ─── Step 4: Sourcer ───
print("=" * 70)
print("STEP 4: SOURCER")
print("=" * 70)
state = make_distilled_context(state)
print()

# ─── Step 5 & 6: Writer ↔ Reviewer Loop ───
print("=" * 70)
print("STEP 5-6: WRITER ↔ REVIEWER (iterative)")
print("=" * 70)

for cycle in range(MAX_REVIEW_CYCLES):
    print(f"\n--- Cycle {cycle + 1}/{MAX_REVIEW_CYCLES} ---")

    # Writer pass
    state = news_writer(state)

    # Reviewer pass
    state = review_writer(state)

    # Check if all items are approved or escalated
    pending = [item for item in state["items"] if item.get("reviewer_status") == "REJECTED"]
    approved = [item for item in state["items"] if item.get("reviewer_status") == "APPROVED"]

    print(f"\n  Approved: {len(approved)} | Rejected (pending rewrite): {len(pending)}")

    if not pending:
        print("  All items approved. Exiting loop.")
        break

    # Mark remaining rejected items for DLQ if this is the last cycle
    if cycle == MAX_REVIEW_CYCLES - 1:
        for item in pending:
            item["status"] = "escalated"
            print(f"  [DLQ] Escalated: {item['title'][:50]}")

print()

# ─── Final Output ───
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

approved_items = [item for item in state["items"] if item.get("reviewer_status") == "APPROVED"]
escalated_items = [item for item in state["items"] if item.get("status") == "escalated"]

print(f"\nTotal processed: {len(state['items'])}")
print(f"Approved: {len(approved_items)}")
print(f"Escalated (DLQ): {len(escalated_items)}")
print(f"Duplicates removed: {state['duplicates_removed']}")

print(f"\n{'─'*70}")
print("APPROVED ARTICLES:")
print(f"{'─'*70}")
for i, item in enumerate(approved_items):
    print(f"\n[{i+1}] {item['title'][:60]}")
    print(f"    Class: {item['news_class']} | Reviewer: {item['reviewer_reasoning'][:80]}")
    if item.get("writer_draft"):
        print(f"    Draft preview: {item['writer_draft']['full_content'][:150]}...")

if escalated_items:
    print(f"\n{'─'*70}")
    print("ESCALATED (DLQ):")
    print(f"{'─'*70}")
    for item in escalated_items:
        print(f"  ✗ {item['title'][:60]}")
        print(f"    Reason: {item.get('reviewer_reasoning', 'Unknown')}")
