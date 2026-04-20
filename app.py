import streamlit as st
import os
import tempfile
import logging
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path
import json

from main_rag import (
    get_hybrid_retriever_from_csv,
    run_rag_pipeline,
    EvidenceAttributor,
    TokenAttributor,
    CitationMetrics
)
from ingestion import update_knowledge_base, deduplicate_sources
from langchain_ollama import OllamaLLM

# Configuration
CSV_PATH = "educational_knowledge_base.csv"
ASSETS_DIR = Path("project_data/assets")

st.set_page_config(
    page_title="EduRAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background-color: #0b0f19 !important;
        background-image: radial-gradient(circle at 50% 0%, #1e293b 0%, #0b0f19 70%);
    }
    .main { background-color: transparent !important; }
    .block-container {
        background-color: transparent !important;
        padding: 2rem 2rem !important;
        max-width: 1400px !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] > div { background-color: transparent !important; }

    p, span, label, h1, h2, h3, h4, h5, h6, div { color: #e2e8f0 !important; }
    h1 { color: #38bdf8 !important; font-weight: 700; font-size: 1.8em !important; }
    h2, h3 { color: #60a5fa !important; font-weight: 600; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e293b !important;
        border-radius: 8px !important;
        padding: 6px !important;
        border: 1px solid #334155 !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        background-color: transparent !important;
        border: none !important;
        font-weight: 500 !important;
        padding: 8px 20px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        background-color: #0f172a !important;
        border-radius: 6px !important;
        border: 1px solid #334155 !important;
    }

    /* Inputs */
    div[data-baseweb="input"] > div,
    .stTextInput > div > div,
    .stTextArea > div > div {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"] > div:focus-within,
    .stTextInput > div > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }
    input, textarea { color: #f1f5f9 !important; }
    input::placeholder { color: #64748b !important; }

    /* Selectbox */
    div[data-baseweb="select"] > div,
    .stSelectbox > div > div {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        color: #f1f5f9 !important;
    }

    /* File uploader */
    .stFileUploader > div > div {
        background-color: #1e293b !important;
        border: 2px dashed #475569 !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }
    .stFileUploader > div > div:hover { border-color: #3b82f6 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.45) !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
    }
    div[data-testid="stMetric"] > label {
        color: #94a3b8 !important;
        font-size: 0.8em !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    .streamlit-expanderContent {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #0f2444 0%, #1e3a8a 100%) !important;
        border-left: 4px solid #60a5fa !important;
        border-radius: 12px !important;
        padding: 24px 28px !important;
        margin: 12px 0 !important;
        box-shadow: 0 8px 24px rgba(30, 58, 138, 0.25) !important;
    }
    .answer-box p {
        color: #f0f9ff !important;
        font-size: 1.05em !important;
        line-height: 1.75 !important;
        margin: 0 !important;
    }

    /* Source cards */
    .source-card {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        transition: border-color 0.2s ease !important;
    }
    .source-card:hover { border-color: #3b82f6 !important; }

    /* Attribution bars */
    .attribution-bar {
        height: 8px !important;
        background-color: #0f172a !important;
        border-radius: 4px !important;
        overflow: hidden !important;
        margin-top: 8px !important;
    }
    .attribution-fill {
        height: 100% !important;
        background: linear-gradient(90deg, #2563eb, #38bdf8) !important;
        border-radius: 4px !important;
    }

    /* Token heatmap */
    .token-grounded {
        background-color: #052e16 !important;
        color: #86efac !important;
        padding: 3px 7px !important;
        border-radius: 5px !important;
        font-weight: 500 !important;
        border: 1px solid #16a34a !important;
        font-size: 0.9em !important;
    }
    .token-ungrounded {
        background-color: #450a0a !important;
        color: #fca5a5 !important;
        padding: 3px 7px !important;
        border-radius: 5px !important;
        font-weight: 500 !important;
        border: 1px solid #dc2626 !important;
        font-size: 0.9em !important;
    }
    .token-partial {
        background-color: #431407 !important;
        color: #fdba74 !important;
        padding: 3px 7px !important;
        border-radius: 5px !important;
        font-weight: 500 !important;
        border: 1px solid #ea580c !important;
        font-size: 0.9em !important;
    }

    /* Reliability badges */
    .badge-success {
        background: linear-gradient(135deg, #059669, #10b981) !important;
        color: #fff !important;
        padding: 6px 14px !important;
        border-radius: 20px !important;
        font-size: 0.85em !important;
        font-weight: 700 !important;
        display: inline-block !important;
    }
    .badge-warning {
        background: linear-gradient(135deg, #b45309, #d97706) !important;
        color: #fff !important;
        padding: 6px 14px !important;
        border-radius: 20px !important;
        font-size: 0.85em !important;
        font-weight: 700 !important;
        display: inline-block !important;
    }
    .badge-danger {
        background: linear-gradient(135deg, #b91c1c, #dc2626) !important;
        color: #fff !important;
        padding: 6px 14px !important;
        border-radius: 20px !important;
        font-size: 0.85em !important;
        font-weight: 700 !important;
        display: inline-block !important;
    }

    /* Chat history */
    .chat-item-user {
        background-color: #1e3a5f !important;
        border: 1px solid #2563eb !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        margin-bottom: 8px !important;
    }
    .chat-item-assistant {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        margin-bottom: 16px !important;
    }

    /* Stat card */
    .stat-card {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 16px !important;
        text-align: center !important;
    }

    /* Dataframe */
    .stDataFrame { background-color: #1e293b !important; }
    .stDataFrame td, .stDataFrame th {
        color: #e2e8f0 !important;
        border-color: #334155 !important;
        background-color: #1e293b !important;
    }

    figcaption {
        color: #94a3b8 !important;
        background-color: #1e293b !important;
        padding: 6px !important;
        border-radius: 0 0 8px 8px !important;
        font-size: 0.8em !important;
    }

    hr { border-color: #1e293b !important; margin: 1.5rem 0 !important; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0f172a !important; }
    ::-webkit-scrollbar-thumb { background: #334155 !important; border-radius: 4px !important; }
    ::-webkit-scrollbar-thumb:hover { background: #475569 !important; }

    .stCode pre {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
    }

    /* Divider line in sidebar */
    .sidebar-divider {
        border-top: 1px solid #334155;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Session State ====================
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_attribution' not in st.session_state:
    st.session_state.show_attribution = True
if 'show_tokens' not in st.session_state:
    st.session_state.show_tokens = True
if 'evidence_attributor' not in st.session_state:
    st.session_state.evidence_attributor = None
if 'token_attributor' not in st.session_state:
    st.session_state.token_attributor = None

# ==================== Pipeline Loader ====================
@st.cache_resource
def load_pipeline():
    """Load retriever and attributors once. Cached across reruns."""
    try:
        retriever = get_hybrid_retriever_from_csv(CSV_PATH, top_k=5, enable_hyde=True)
        evidence_attributor = EvidenceAttributor()
        token_attributor = TokenAttributor()
        return retriever, evidence_attributor, token_attributor
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        return None, None, None

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown("## 🎓 EduRAG")
    st.caption("Explainable Educational Q&A")
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Knowledge Base Stats
    st.markdown("#### 📊 Knowledge Base")
    if os.path.exists(CSV_PATH):
        try:
            df_stats = pd.read_csv(CSV_PATH)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Sources", len(df_stats['source_file'].unique()))
                st.metric("Text", len(df_stats[df_stats['type'] == 'text']))
            with c2:
                st.metric("Chunks", len(df_stats))
                st.metric("Visuals", len(df_stats[df_stats['type'] == 'visual_content']))

            with st.expander("📁 Sources"):
                for src in df_stats['source_file'].unique()[:6]:
                    st.markdown(f"<div style='font-size:0.8em; color:#94a3b8; padding:2px 0;'>📄 {src[:35]}{'...' if len(src)>35 else ''}</div>", unsafe_allow_html=True)
                extra = len(df_stats['source_file'].unique()) - 6
                if extra > 0:
                    st.caption(f"... and {extra} more")
        except Exception as e:
            st.warning(f"Error reading KB: {e}")
    else:
        st.warning("⚠️ No knowledge base found.\nUpload documents first.")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Display settings
    st.markdown("#### ⚙️ Display Settings")
    st.session_state.show_attribution = st.toggle(
        "Show Evidence Attribution", value=st.session_state.show_attribution
    )
    st.session_state.show_tokens = st.toggle(
        "Show Token Grounding", value=st.session_state.show_tokens
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Chat history management
    st.markdown("#### 💬 Chat History")
    st.caption(f"{len(st.session_state.chat_history)} conversation(s)")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # System info
    st.markdown("#### 🤖 System")
    st.markdown("""
    <div style='font-size:0.82em; color:#64748b; line-height:1.8;'>
    🔵 LLM: DeepSeek R1 8B<br>
    🟢 Embeddings: BGE-small<br>
    🟡 Reranker: MS-MARCO<br>
    🔒 Local Processing
    </div>
    """, unsafe_allow_html=True)

# ==================== Load Pipeline ====================
retriever, evidence_attributor, token_attributor = load_pipeline()

# Store attributors in session for reuse
if retriever and st.session_state.evidence_attributor is None:
    st.session_state.evidence_attributor = evidence_attributor
    st.session_state.token_attributor = token_attributor

# ==================== Tabs ====================
qa_tab, upload_tab, viz_tab = st.tabs(["📚 QA Assistant", "⬆️ Upload Sources", "🔍 Knowledge Map"])

# ====================================================
# TAB 1: QA Assistant
# ====================================================
with qa_tab:
    st.markdown("## 📚 Educational Q&A Assistant")
    st.caption("Ask questions grounded in your uploaded documents — with citations and explainability.")

    if not retriever:
        st.error("❌ System not loaded. Check that Ollama is running and a knowledge base exists.")
    else:
        # Query input
        col_q, col_mode = st.columns([4, 1])
        with col_q:
            user_query = st.text_input(
                "Your question:",
                placeholder="e.g. Explain the attention mechanism in transformers",
                label_visibility="collapsed"
            )
        with col_mode:
            search_type = st.selectbox(
                "Mode",
                ["Auto", "Text Only", "Visual/Diagrams"],
                label_visibility="collapsed"
            )

        submit = st.button("🔍 Ask", type="primary", use_container_width=False)

        if submit and user_query.strip():
            query_input = f"[VISUAL] {user_query}" if search_type == "Visual/Diagrams" else user_query

            with st.spinner("Retrieving and generating answer..."):
                try:
                    result = run_rag_pipeline(
                        query_input,
                        retriever,
                        llm=None,
                        evidence_attributor=st.session_state.evidence_attributor,
                        token_attributor=st.session_state.token_attributor
                    )

                    # Save to chat history
                    st.session_state.chat_history.append({
                        "query": user_query,
                        "answer": result.answer,
                        "faithfulness": result.faithfulness,
                        "grounding": result.token_attribution.get('grounding_ratio', 0)
                    })

                    # ---- Answer ----
                    st.markdown("### 💡 Answer")

                    ans_col, meta_col = st.columns([3, 1])

                    with ans_col:
                        st.markdown(f"""
                        <div class="answer-box">
                            <p>{result.answer}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"Based on {len(result.context)} retrieved sources")

                    with meta_col:
                        reliability = (
                            result.faithfulness +
                            result.retrieval_score +
                            result.consensus.get('consensus_score', 0)
                        ) / 3

                        if reliability > 0.75:
                            badge = '<div class="badge-success">✅ High Reliability</div>'
                        elif reliability > 0.55:
                            badge = '<div class="badge-warning">⚠️ Moderate</div>'
                        else:
                            badge = '<div class="badge-danger">❌ Low — Verify</div>'

                        st.markdown(badge, unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.metric("Faithfulness", f"{result.faithfulness:.2f}")

                        # FIX: Handle None citation rate
                        cit_rate = result.citation_metrics.get('valid_citation_rate')
                        cit_display = f"{cit_rate * 100:.0f}%" if cit_rate is not None else "—"
                        st.metric("Citation Valid", cit_display)

                    # ---- Reliability Metrics ----
                    with st.expander("📊 Reliability Metrics", expanded=True):
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Retrieval Score", f"{result.retrieval_score:.3f}",
                                      help="Average cosine similarity of retrieved chunks to query")
                        with m2:
                            st.metric("Faithfulness", f"{result.faithfulness:.3f}",
                                      help="BERTScore between answer and best-matching retrieved chunk")
                        with m3:
                            con = result.consensus.get('consensus_score', 0)
                            st.metric("Source Consensus", f"{con:.3f}",
                                      help="How much retrieved sources agree with each other")
                        with m4:
                            gr = result.token_attribution.get('grounding_ratio', 0)
                            st.metric("Token Grounding", f"{gr * 100:.1f}%",
                                      help="% of content words in answer grounded in retrieved context (stop words excluded)")

                        if result.consensus.get('status') == "CONFLICT_DETECTED":
                            st.warning("⚠️ Conflicting sources detected — verify claims independently.")
                            for conflict in result.consensus.get('conflicts', [])[:2]:
                                st.caption(f"Conflict: {conflict['doc_a']['id']} ↔ {conflict['doc_b']['id']} (similarity: {conflict['similarity']:.2f})")

                    # ---- Evidence Attribution ----
                    if st.session_state.show_attribution and result.evidence_attribution.get('attributions'):
                        st.markdown("### 🔍 Evidence Attribution")
                        st.caption("Which sources contributed most to this answer")

                        attrs = result.evidence_attribution['attributions'][:3]
                        cols = st.columns(len(attrs))

                        for idx, attr in enumerate(attrs):
                            with cols[idx]:
                                pct = attr['contribution_pct']
                                src_short = attr['source'][:22] + "..." if len(attr['source']) > 22 else attr['source']
                                st.markdown(f"""
                                <div style="background:#0f172a; padding:14px; border-radius:8px; border:1px solid #334155;">
                                    <div style="font-weight:600; color:#38bdf8; font-size:0.9em; margin-bottom:4px;">
                                        📄 {src_short}
                                    </div>
                                    <div style="font-size:0.78em; color:#64748b; margin-bottom:8px;">
                                        Page {attr['page']}
                                    </div>
                                    <div class="attribution-bar">
                                        <div class="attribution-fill" style="width:{pct}%;"></div>
                                    </div>
                                    <div style="text-align:right; color:#38bdf8; font-weight:700; font-size:0.9em; margin-top:4px;">
                                        {pct:.1f}%
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                    # ---- Token Grounding ----
                    if st.session_state.show_tokens and result.token_attribution.get('tokens'):
                        st.markdown("### 🔤 Token Grounding Heatmap")
                        st.caption("🟢 Grounded in context &nbsp;|&nbsp; 🟠 Partial &nbsp;|&nbsp; 🔴 Ungrounded (potential hallucination)")

                        html = '<div style="line-height:2.4; word-spacing:4px;">'
                        for t in result.token_attribution['tokens'][:80]:
                            token = t['token']
                            score = t['grounded_score']
                            if score > 0.8:
                                cls = "token-grounded"
                            elif score > 0.5:
                                cls = "token-partial"
                            else:
                                cls = "token-ungrounded"
                            html += f'<span class="{cls}">{token}</span> '
                        html += '</div>'

                        st.markdown(f"""
                        <div style="background:#0f172a; padding:20px; border-radius:10px; border:1px solid #334155;">
                            {html}
                        </div>
                        """, unsafe_allow_html=True)

                        ungrounded = result.token_attribution.get('ungrounded_tokens', [])
                        if ungrounded:
                            st.caption(f"⚠️ Potentially ungrounded content words: {', '.join(ungrounded[:8])}")

                    # ---- Sources ----
                    st.markdown("### 📚 Retrieved Sources")

                    text_docs = [d for d in result.context if d.metadata.get("type") != "visual_content"]
                    visual_docs = [d for d in result.context if d.metadata.get("type") == "visual_content"]

                    if visual_docs:
                        st.markdown("#### 🖼️ Visual References")
                        img_cols = st.columns(min(len(visual_docs), 3))
                        for idx, doc in enumerate(visual_docs):
                            with img_cols[idx % 3]:
                                img_path = doc.metadata.get("image_ref")
                                if img_path and os.path.exists(str(img_path)):
                                    st.image(
                                        str(img_path),
                                        caption=f"Page {doc.metadata.get('page_number')} · {doc.metadata.get('source_file', '')[:20]}"
                                    )
                                else:
                                    # Try alternative path resolution
                                    alt_path = os.path.join(os.getcwd(), str(img_path)) if img_path else ""
                                    if alt_path and os.path.exists(alt_path):
                                        st.image(alt_path, caption=f"Page {doc.metadata.get('page_number')}")
                                    else:
                                        st.warning(f"Image not found: {img_path}")

                    if text_docs:
                        st.markdown("#### 📝 Text Sources")
                        for i, doc in enumerate(text_docs, 1):
                            with st.expander(
                                f"[{i}] {doc.metadata.get('source_file', 'Unknown')} · Page {doc.metadata.get('page_number', 'N/A')} · {doc.metadata.get('section_path', 'General')}"
                            ):
                                st.markdown(f"""
                                <div style="color:#cbd5e1; font-size:0.93em; line-height:1.65;">
                                {doc.page_content[:500]}{"..." if len(doc.page_content) > 500 else ""}
                                </div>
                                """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # ---- Chat History ----
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 💬 Previous Questions")
            for i, item in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                st.markdown(f"""
                <div class="chat-item-user">
                    <span style="font-size:0.8em; color:#64748b;">Q{len(st.session_state.chat_history)-i+1}</span><br>
                    <span style="font-weight:600; color:#e2e8f0;">{item['query']}</span>
                </div>
                <div class="chat-item-assistant">
                    <span style="font-size:0.78em; color:#64748b;">Answer · Faithfulness: {item['faithfulness']:.2f} · Grounding: {item['grounding']*100:.0f}%</span><br>
                    <span style="color:#cbd5e1; font-size:0.9em;">{item['answer'][:200]}{"..." if len(item['answer'])>200 else ""}</span>
                </div>
                """, unsafe_allow_html=True)

# ====================================================
# TAB 2: Upload Sources
# ====================================================
with upload_tab:
    st.markdown("## ⬆️ Knowledge Base Management")
    st.caption("Add PDFs, Word documents, or PowerPoint files to your knowledge base.")

    # Settings
    with st.expander("⚙️ Advanced Options"):
        vlm_enabled = st.toggle(
            "Enable VLM Image Description (LLaVA-Phi3)",
            value=True,
            help="Uses a vision model to generate searchable text from diagrams. Requires Ollama with llava-phi3. Slower but more complete."
        )
        st.info("💡 Disable VLM for faster processing if your documents have no important diagrams.")

    # Upload area
    uploaded_files = st.file_uploader(
        "Upload Documents:",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        help="Supported formats: PDF, Word (.docx), PowerPoint (.pptx)"
    )

    url_input = st.text_area(
        "Or enter URLs (one per line):",
        placeholder="https://example.com/textbook.pdf",
        height=80
    )

    # Summary of what will be processed
    sources_preview = []
    if uploaded_files:
        sources_preview += [f.name for f in uploaded_files]
    if url_input.strip():
        sources_preview += [u.strip() for u in url_input.split('\n') if u.strip()]

    if sources_preview:
        st.markdown(f"**Ready to process {len(sources_preview)} source(s):**")
        for s in sources_preview[:5]:
            st.caption(f"• {s}")
        if len(sources_preview) > 5:
            st.caption(f"... and {len(sources_preview) - 5} more")

    if st.button("⚡ Process & Update Knowledge Base", type="primary", use_container_width=False):
        if not sources_preview:
            st.warning("⚠️ No sources provided. Upload files or enter URLs.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                sources_to_process = []

                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(tmpdir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        sources_to_process.append(temp_path)

                if url_input.strip():
                    sources_to_process.extend([u.strip() for u in url_input.split('\n') if u.strip()])

                progress_bar = st.progress(0, text="Starting...")
                status_text = st.empty()

                try:
                    for i, src in enumerate(sources_to_process):
                        name = os.path.basename(src) if os.path.exists(src) else src
                        status_text.markdown(f"Processing: **{name}**")
                        progress_bar.progress((i + 1) / len(sources_to_process), text=f"Processing {i+1}/{len(sources_to_process)}...")

                    total_chunks = update_knowledge_base(
                        sources_to_process,
                        kb_path=CSV_PATH,
                        assets_dir=str(ASSETS_DIR),
                        vlm_enabled=vlm_enabled
                    )

                    progress_bar.progress(1.0, text="Complete!")
                    status_text.empty()
                    load_pipeline.clear()

                    st.success(f"✅ Done! Knowledge base now contains **{total_chunks} chunks**.")
                    st.info("🔄 Refresh the page to use the updated knowledge base.")
                    st.balloons()

                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    st.exception(e)

# ====================================================
# TAB 3: Knowledge Map
# ====================================================
with viz_tab:
    st.markdown("## 🔍 Knowledge Base Explorer")
    st.caption("Browse documents, sections, and chunks in your knowledge base.")

    if not os.path.exists(CSV_PATH):
        st.info("📂 No knowledge base found. Upload documents to get started.")
    else:
        try:
            df = pd.read_csv(CSV_PATH)

            # Summary stats row
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Total Documents", len(df['source_file'].unique()))
            with s2:
                st.metric("Total Chunks", len(df))
            with s3:
                st.metric("Text Chunks", len(df[df['type'] == 'text']))
            with s4:
                st.metric("Visual Chunks", len(df[df['type'] == 'visual_content']))

            st.markdown("---")

            # Filter controls
            f1, f2 = st.columns([1, 1])
            with f1:
                sources = df['source_file'].unique().tolist()
                selected_source = st.selectbox("📄 Filter by Document:", ["All Documents"] + sources)

            filtered_df = df if selected_source == "All Documents" else df[df['source_file'] == selected_source]

            with f2:
                sections = filtered_df['section_path'].unique().tolist()
                selected_section = st.selectbox("📑 Filter by Section:", ["All Sections"] + sections[:30])

            if selected_section != "All Sections":
                filtered_df = filtered_df[filtered_df['section_path'] == selected_section]

            st.caption(f"Showing **{len(filtered_df)}** chunks")

            # Content type tabs
            text_df = filtered_df[filtered_df['type'] != 'visual_content']
            visual_df = filtered_df[filtered_df['type'] == 'visual_content']

            content_tab1, content_tab2 = st.tabs([
                f"📝 Text & Tables ({len(text_df)})",
                f"🖼️ Visual Content ({len(visual_df)})"
            ])

            with content_tab1:
                if text_df.empty:
                    st.info("No text chunks in this selection.")
                else:
                    for _, row in text_df.head(20).iterrows():
                        chunk_id = row.get('chunk_id', 'Unknown')
                        page = row.get('page_number', 'N/A')
                        section = row.get('section_path', 'General')
                        content = str(row.get('text', ''))
                        chunk_type = row.get('type', 'text')

                        type_badge = "📊 Table" if chunk_type == 'table' else "📄 Text"

                        with st.expander(f"{type_badge} · {chunk_id} · Page {page}"):
                            st.caption(f"Section: {section}")
                            st.markdown(f"""
                            <div style="background:#0f172a; padding:14px; border-radius:8px; color:#cbd5e1; font-size:0.92em; line-height:1.65; white-space:pre-wrap;">
{content[:600]}{"..." if len(content) > 600 else ""}
                            </div>
                            """, unsafe_allow_html=True)

                    if len(text_df) > 20:
                        st.caption(f"Showing first 20 of {len(text_df)} chunks.")

            with content_tab2:
                if visual_df.empty:
                    st.info("No visual content in this selection.")
                else:
                    img_cols = st.columns(3)
                    for i, (_, row) in enumerate(visual_df.head(12).iterrows()):
                        with img_cols[i % 3]:
                            img_path = row.get('image_ref', '')
                            if img_path and os.path.exists(str(img_path)):
                                st.image(
                                    str(img_path),
                                    caption=f"Page {row.get('page_number', 'N/A')} · {row.get('source_file', '')[:20]}"
                                )
                                desc = str(row.get('text', ''))
                                if desc:
                                    st.caption(desc[:100])
                            else:
                                st.markdown(f"""
                                <div style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:20px; text-align:center; color:#64748b;">
                                    🖼️ Image not found<br>
                                    <span style="font-size:0.8em;">{row.get('chunk_id', '')}</span>
                                </div>
                                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
            import traceback
            st.code(traceback.format_exc())

# ==================== Footer ====================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.8em;">
    EduRAG · Docling + Ollama + LangChain + FAISS · Local Processing · Privacy First
</div>
""", unsafe_allow_html=True)