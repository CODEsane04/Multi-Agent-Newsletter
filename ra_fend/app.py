import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ra_bend"))

import streamlit as st
from agents.scout import scout_global_state_update
from agents.classifier import classify_news
from agents.sourcer import make_distilled_context
from agents.writer import news_writer
from agents.reviewer import review_writer

st.set_page_config(page_title="AI Newsletter", layout="wide", initial_sidebar_state="collapsed")

# ─── Dark theme CSS ───
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .news-title {
        font-size: 1.1rem;
        padding: 12px 16px;
        margin: 4px 0;
        border-radius: 6px;
        background: #1a1f2e;
        border-left: 3px solid #4a9eff;
        cursor: pointer;
        transition: background 0.2s;
    }
    .news-title:hover { background: #252b3b; }
    .class-badge {
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 10px;
        background: #2a3a5a;
        color: #7ab3ff;
        margin-left: 8px;
    }
    .draft-content {
        background: #1a1f2e;
        padding: 20px;
        border-radius: 8px;
        margin-top: 10px;
        line-height: 1.7;
    }
    .summary-box {
        background: #162030;
        padding: 16px;
        border-radius: 8px;
        border-left: 3px solid #34d399;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───
st.title("🗞️ AI Research Newsletter")
st.caption("Auto-curated AI news — click any headline to read the full draft")

# ─── Run pipeline (cached) ───
@st.cache_data(show_spinner="Running pipeline...")
def run_pipeline():
    state = scout_global_state_update()
    state["items"] = state["items"][:8]  # Limit for speed
    state = classify_news(state)
    state = make_distilled_context(state)
    state = news_writer(state)
    state = review_writer(state)
    return state

# Sidebar: trigger pipeline
with st.sidebar:
    if st.button("🔄 Refresh News", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

state = run_pipeline()
items = [item for item in state["items"] if item.get("writer_draft")]

# ─── News list ───
if not items:
    st.info("No news items processed yet. Click Refresh.")
else:
    # Track selected item
    if "selected" not in st.session_state:
        st.session_state.selected = None

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Headlines")
        for i, item in enumerate(items):
            class_labels = {1: "Model", 2: "Research", 3: "Product", 4: "Hardware", 5: "Policy", 6: "General"}
            label = class_labels.get(item["news_class"], "General")

            if st.button(f"{item['title'][:70]}", key=f"btn_{i}", use_container_width=True):
                st.session_state.selected = i

    with col2:
        if st.session_state.selected is not None:
            item = items[st.session_state.selected]
            draft = item["writer_draft"]

            st.subheader(item["title"])
            st.caption(f"Source: {item['source']} | Class: {item['news_class']} | Status: {item.get('reviewer_status', 'N/A')}")

            st.markdown("---")
            st.markdown(draft["full_content"])

            st.markdown("---")
            st.markdown("**📋 Key Claims (6-point summary)**")
            st.markdown(f'<div class="summary-box">{draft["summary"]}</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"[🔗 Read original article]({item['url']})")
        else:
            st.markdown("*← Click a headline to read the full article*")
