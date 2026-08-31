"""
Test: Scout → Classifier pipeline (full run, batched)
"""
import sys
sys.path.insert(0, ".")

from agents.scout import scout_global_state_update
from agents.classifier import classify_news

# Fetch items
state = scout_global_state_update()
print(f"Fetched {len(state['items'])} items\n")

# Classify
state = classify_news(state)

# Print results
print(f"\n{'='*70}")
print(f"CLASSIFICATION RESULTS")
print(f"{'='*70}")
for item in state["items"]:
    print(f"  [{item['news_class'][:30]}] {item['title'][:55]}")
