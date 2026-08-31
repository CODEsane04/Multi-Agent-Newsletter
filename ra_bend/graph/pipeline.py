# Main LangGraph DAG — nodes + edges wiring
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, START, END
from graph.state import NewsItem, PipelineState
from agents.scout import scout_global_state_update
from agents.dedup import remove_duplicates
from agents.classifier import classify_news
from agents.sourcer import make_distilled_context
from agents.writer import news_writer
from agents.reviewer import review_writer
from agents.route_evaluator import route_evaluation

## define the graph
graph = StateGraph(PipelineState)

## Add nodes to the graph
graph.add_node('scout_agent', scout_global_state_update)
graph.add_node('dedup_agent', remove_duplicates)
graph.add_node('classifier_agent', classify_news)
graph.add_node('sourcer_agent', make_distilled_context)
graph.add_node('writer_agent', news_writer)
graph.add_node('reviewer_agent', review_writer)

## Add edges to the graph
graph.add_edge(START, 'scout_agent') #sequential
graph.add_edge('scout_agent', 'dedup_agent') #sequential
graph.add_edge('dedup_agent', 'classifier_agent') #sequential
graph.add_edge('classifier_agent', 'sourcer_agent') #sequential
graph.add_edge('sourcer_agent', 'writer_agent') #sequential

## Iterative work-flow
graph.add_edge('writer_agent', 'reviewer_agent')
graph.add_conditional_edges('reviewer_agent', route_evaluation, {'approved' : END, 're-write' : 'writer_agent'})

## compile the graph
workflow = graph.compile()

## execute the graph
initial_state = {
    "run_id": "",
    "triggered_at": "",
    "items": [],
    "all_approved": False,
    "iterations": 0,
    "max_iterations": 4,
    "total_signals_fetched": 0,
    "duplicates_removed": 0,
    "items_filtered_out": 0,
    "total_cost_usd": 0.0,
    "run_status": "running",
}

final_state = workflow.invoke(initial_state)

## Display results
print("\n" + "=" * 70)
print("📰 NEWSLETTER DRAFTS")
print("=" * 70)

for i, item in enumerate(final_state["items"]):
    if item.get("writer_draft"):
        print(f"\n{'─' * 70}")
        print(f"[{i+1}] {item['title']}")
        print(f"    Class: {item['news_class']} | Status: {item['reviewer_status']}")
        print(f"{'─' * 70}")
        print(item["writer_draft"]["full_content"])
        print(f"\n📋 Summary:\n{item['writer_draft']['summary']}")
    else:
        print(f"\n[{i+1}] {item['title']} — [NO DRAFT]")

print(f"\n{'=' * 70}")
print(f"Pipeline complete. Items: {len(final_state['items'])} | Iterations: {final_state['iterations']} | All approved: {final_state['all_approved']}")
print(f"{'=' * 70}")