"""
Greenbook Assistant — Streamlit Frontend
Connects to the FastAPI backend at /rag for PG&E document Q&A.
"""

import streamlit as st
import requests
import time
from datetime import datetime
import markdown
import base64
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = "http://localhost:8000"
RAG_ENDPOINT = f"{BACKEND_URL}/rag"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"
IMAGE_ENDPOINT = f"{BACKEND_URL}/images"
DEFAULT_MODEL = "groq/llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PG&E Greenbook Manual Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .header-container {
        background: linear-gradient(135deg, #1a365d 0%, #2d5a8e 50%, #1e4976 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(26, 54, 93, 0.3);
    }
    .header-title {
        color: white;
        font-size: 1.6rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-badge {
        background: rgba(255,255,255,0.15);
        color: #e2e8f0;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }

    /* ── Chat Messages ── */
    .user-msg {
        background: linear-gradient(135deg, #2d5a8e, #1a365d);
        color: white;
        padding: 14px 20px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(26,54,93,0.2);
    }
    .bot-msg {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        padding: 18px 22px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        font-size: 0.95rem;
        line-height: 1.7;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .bot-msg img {
        max-width: 600px;
        width: 100%;
        object-fit: contain;
        border-radius: 8px;
        margin: 12px auto;
        display: block;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .img-caption {
        text-align: center;
        font-size: 0.78rem;
        color: #718096;
        margin: 4px 0 12px 0;
        font-style: italic;
    }

    /* ── Markdown tables inside bot messages ── */
    .bot-msg table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        margin: 12px 0;
    }
    .bot-msg table th {
        background: #1a365d;
        color: white;
        padding: 8px 12px;
        text-align: left;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .bot-msg table td {
        padding: 7px 12px;
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.82rem;
    }
    .bot-msg table tr:nth-child(even) {
        background: #f0f4f8;
    }
    .bot-msg table tr:hover {
        background: #e8f0fe;
    }

    /* ── Sources ── */
    .source-pill {
        display: inline-block;
        background: #e8f0fe;
        color: #1a365d;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 3px 4px 3px 0;
        border: 1px solid #bee3f8;
    }

    /* ── Stats bar ── */
    .stats-bar {
        display: flex;
        gap: 16px;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #e2e8f0;
        font-size: 0.75rem;
        color: #718096;
        flex-wrap: wrap;
    }
    .stat-item {
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* ── Status indicators ── */
    .status-online {
        color: #38a169;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-offline {
        color: #e53e3e;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* ── Sidebar styling ── */
    .sidebar-section {
        background: #f7fafc;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 12px;
        border: 1px solid #e2e8f0;
    }
    .sidebar-section h4 {
        margin: 0 0 8px 0;
        font-size: 0.85rem;
        color: #2d3748;
    }

    /* ── Table styling ── */
    .data-table-wrap {
        margin: 14px 0;
        overflow-x: auto;
    }
    .data-table-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #1a365d;
        margin-bottom: 6px;
    }
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .data-table th {
        background: #1a365d;
        color: white;
        padding: 8px 12px;
        text-align: left;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .data-table td {
        padding: 7px 12px;
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.82rem;
    }
    .data-table tr:nth-child(even) {
        background: #f0f4f8;
    }
    .data-table tr:hover {
        background: #e8f0fe;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def check_backend_health():
    """Check if the backend API is reachable."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def query_rag(question: str, model: str) -> dict:
    """Send a query to the /rag endpoint."""
    payload = {
        "query": question,
        "model": model,
    }
    try:
        resp = requests.post(RAG_ENDPOINT, json=payload, timeout=60)
        if resp.status_code == 429:
            return {"status": "error", "answer": "⏳ **Rate limit reached.** Please switch to a different model in the sidebar (e.g., `mixtral-8x7b`) or wait a few minutes and try again."}
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "answer": "❌ Cannot connect to backend. Is the API server running on port 8000?"}
    except requests.exceptions.Timeout:
        return {"status": "error", "answer": "⏳ Request timed out. The query may be too complex."}
    except requests.exceptions.HTTPError as e:
        if "429" in str(e) or "rate" in str(e).lower():
            return {"status": "error", "answer": "⏳ **Rate limit reached.** Please switch to a different model in the sidebar or wait a few minutes."}
        return {"status": "error", "answer": f"❌ Error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "answer": f"❌ Error: {str(e)}"}


def fetch_image_b64(image_id: str) -> str | None:
    """Fetch image from backend and return as base64 data URI."""
    try:
        resp = requests.get(f"{IMAGE_ENDPOINT}/{image_id}", timeout=10)
        if resp.status_code == 200:
            b64 = base64.b64encode(resp.content).decode("utf-8")
            return f"data:image/png;base64,{b64}"
    except Exception:
        pass
    return None


def format_sources_html(sources: list) -> str:
    """Render source pills as HTML."""
    if not sources:
        return ""
    pills = "".join(
        f'<span class="source-pill">📄 {s.get("title", "")} </span>' for s in sources
    )
    return f'<div style="margin-top:10px">{pills}</div>'


def format_stats_html(meta: dict) -> str:
    """Render stats bar as HTML."""
    model = meta.get("modelused", "")
    model_display = model.split("/")[-1] if "/" in model else model
    total_ms = meta.get("totaltimems", 0)
    total_tokens = meta.get("totaltokens", 0)
    retrieval_ms = meta.get("retrievaltimems", 0)
    gen_ms = meta.get("generationtimems", 0)

    return f"""
    <div class="stats-bar">
        <span class="stat-item">🤖 {model_display}</span>
        <span class="stat-item">⏱ {total_ms:,}ms</span>
        <span class="stat-item">📊 {total_tokens:,} tokens</span>
        <span class="stat-item">🔍 {retrieval_ms}ms retrieval</span>
        <span class="stat-item">💡 {gen_ms}ms generation</span>
    </div>
    """


def build_images_html(images: list, cached_b64: dict = None) -> str:
    """
    Build HTML for inline images inside the bot message card.
    Returns HTML string with <img> tags using base64 data URIs.
    
    Images now come as: [{"image_base64": "data:image/png;base64,..."}]
    """
    if not images:
        return ""
    
    html_parts = []
    
    for idx, img in enumerate(images):
        data_uri = img.get("image_base64", "")
        
        if data_uri:
            html_parts.append(
                f'<img src="{data_uri}" alt="Figure {idx + 1}">'
            )
    
    return "".join(html_parts)


def build_tables_html(tables: list) -> str:
    """
    Build HTML for structured tables inside the bot message card.
    
    Args:
        tables: list of table dicts with table_name, page_number, headers, rows
    """
    if not tables:
        return ""
    
    html_parts = []
    
    for tbl in tables:
        name = tbl.get("table_name", "Table")
        page = tbl.get("page_number", "?")
        headers = tbl.get("headers", [])
        rows = tbl.get("rows", [])
        
        if not headers and not rows:
            continue
        
        tbl_html = f'<div class="data-table-wrap">'
        tbl_html += f'<div class="data-table-title">📋 {name} (Page {page})</div>'
        tbl_html += '<table class="data-table">'
        
        # Headers
        if headers:
            tbl_html += '<thead><tr>'
            for h in headers:
                tbl_html += f'<th>{h}</th>'
            tbl_html += '</tr></thead>'
        
        # Rows
        if rows:
            tbl_html += '<tbody>'
            for row in rows:
                tbl_html += '<tr>'
                for cell in row:
                    tbl_html += f'<td>{cell}</td>'
                tbl_html += '</tr>'
            tbl_html += '</tbody>'
        
        tbl_html += '</table></div>'
        html_parts.append(tbl_html)
    
    return "".join(html_parts)


def render_bot_message(answer: str, sources: list, metadata: dict,
                       images: list, tables: list = None,
                       cached_b64: dict = None) -> str:
    """
    Build the full bot message HTML:  text → images → tables → sources → stats.
    All inside a single .bot-msg card.
    """
    html_answer = markdown.markdown(answer, extensions=['extra', 'nl2br'])
    
    bot_html = f'<div class="bot-msg">{html_answer}'
    
    # Inline images
    if images:
        bot_html += build_images_html(images, cached_b64)
    
    # Structured tables
    if tables:
        bot_html += build_tables_html(tables)
    
    # Source pills
    if sources:
        bot_html += format_sources_html(sources)
    
    # Stats bar
    if metadata:
        bot_html += format_stats_html(metadata)
    
    bot_html += "</div>"
    return bot_html


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Model selector
    model_options = [
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
        "groq/llama3-8b-8192",
        "groq/gemma2-9b-it",
    ]
    st.session_state.model = st.selectbox(
        "LLM Model",
        model_options,
        index=0,
    )

    st.divider()

    # Backend status
    st.markdown("### 📡 Backend Status")
    is_online = check_backend_health()
    if is_online:
        st.markdown('<p class="status-online">● Connected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-offline">● Disconnected</p>', unsafe_allow_html=True)
        st.warning("Start backend: `python -m api.main`")

    st.divider()

    # Quick questions
    st.markdown("### 💡 Try these questions")
    sample_questions = [
        "What is the maximum transformer size for single phase service?",
        "What are the load limits for single-phase vs three-phase service?",
        "What conduit size is required for meter wiring?",
        "What are the clearance requirements for underground service?",
        "What is the process to apply for new electric service?",
    ]
    for q in sample_questions:
        if st.button(q, key=f"sample_{hash(q)}", use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="header-container">
    <span class="header-title">📘 Greenbook Assistant</span>
    <span class="header-badge">Graph RAG · PG&E Manual</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat History
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        cached_b64 = msg.get("_image_cache", {})
        bot_html = render_bot_message(
            answer=msg["content"],
            sources=msg.get("sources", []),
            metadata=msg.get("metadata", {}),
            images=msg.get("images", []),
            tables=msg.get("tables", []),
            cached_b64=cached_b64,
        )
        # Persist any newly-fetched base64 data
        msg["_image_cache"] = cached_b64
        st.markdown(bot_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------
pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Ask a question about PG&E Greenbook...")

question = pending or user_input

if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(
        f'<div class="user-msg">{question}</div>', unsafe_allow_html=True
    )

    # Query backend
    with st.spinner("🔍 Searching documents & generating answer..."):
        result = query_rag(question, st.session_state.model)

    # Parse response
    answer = result.get("answer", "No response received.")
    sources = result.get("sources", [])
    metadata = result.get("metadata", {})
    images = result.get("images", [])
    tables = result.get("tables", [])

    # ── Streaming simulation (text only) ──
    placeholder = st.empty()
    words = answer.split(" ")
    streamed = ""
    for word in words:
        streamed += word + " "
        html_partial = markdown.markdown(streamed, extensions=['extra', 'nl2br'])
        placeholder.markdown(
            f'<div class="bot-msg">{html_partial}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.015)

    # ── Final render: text + images + tables + sources + stats ──
    image_cache = {}
    final_html = render_bot_message(
        answer=answer,
        sources=sources,
        metadata=metadata,
        images=images,
        tables=tables,
        cached_b64=image_cache,
    )
    placeholder.markdown(final_html, unsafe_allow_html=True)

    # Store in session
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "metadata": metadata,
        "images": images,
        "tables": tables,
        "_image_cache": image_cache,
    })
