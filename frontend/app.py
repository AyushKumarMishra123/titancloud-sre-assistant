"""
TitanCloud SRE Incident Assistant - Streamlit Frontend Application
Enterprise chatbot interface connecting to FastAPI backend for grounded SRE runbook assistance.
"""

import os
import uuid
from typing import Any, Dict, List, Optional
import requests
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="TitanCloud SRE Incident Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Backend URL configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Enterprise styling (dark theme, crisp typography, clean citation badges)
st.markdown(
    """
    <style>
    /* Global Styles */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Header Card */
    .titan-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 22px 26px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .titan-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .titan-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Citation Card */
    .citation-card {
        background-color: #1a2234;
        border-left: 4px solid #38bdf8;
        border-radius: 4px;
        padding: 10px 14px;
        margin: 8px 0;
    }
    .citation-badge {
        display: inline-block;
        background-color: #0369a1;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        margin-right: 8px;
    }
    
    /* Metric pill */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        font-size: 0.82rem;
        color: #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_backend_health() -> Optional[Dict[str, Any]]:
    """Fetch backend operational status."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_backend_stats() -> Optional[Dict[str, Any]]:
    """Fetch runbook and vector database statistics."""
    try:
        response = requests.get(f"{BACKEND_URL}/stats", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def send_chat_message(
    question: str,
    session_id: str,
    chat_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Sends a user question to the FastAPI /chat endpoint."""
    payload = {
        "question": question,
        "session_id": session_id,
        "chat_history": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in chat_history[-6:]
        ],
        "top_k": 4,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "answer": f"Backend Error (HTTP {response.status_code}): {response.text}",
                "sources": [],
                "retrieval_status": "error",
                "is_sufficient": False,
            }
    except requests.exceptions.ConnectionError:
        return {
            "answer": f"Cannot connect to the backend server at {BACKEND_URL}. Please ensure FastAPI is running.",
            "sources": [],
            "retrieval_status": "error",
            "is_sufficient": False,
        }
    except Exception as e:
        return {
            "answer": f"An unexpected error occurred while communicating with backend: {str(e)}",
            "sources": [],
            "retrieval_status": "error",
            "is_sufficient": False,
        }


def clear_session_history(session_id: str) -> None:
    """Invokes backend to clear session dialogue memory."""
    try:
        requests.post(f"{BACKEND_URL}/chat/clear", json={"session_id": session_id}, timeout=3)
    except Exception:
        pass


# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ TitanCloud SRE")
    st.caption("Enterprise Incident Assistant v1.0")
    st.markdown("---")

    # Fetch live system stats from FastAPI
    health_data = get_backend_health()
    stats_data = get_backend_stats()

    st.markdown("#### System Connectivity")
    if health_data and health_data.get("status") == "healthy":
        st.success("🟢 Backend Connected (Port 8000)")
        gemini_status = "🟢 Configured" if health_data.get("gemini_configured") else "🟡 Offline / Mock Mode"
        st.markdown(f"**Gemini Model**: `{health_data.get('gemini_model', 'N/A')}`")
        st.caption(f"LLM Status: {gemini_status}")
    else:
        st.error(f"🔴 Backend Disconnected ({BACKEND_URL})")
        st.caption("Make sure `uvicorn backend.main:app` is running.")

    st.markdown("---")
    st.markdown("#### Runbook Knowledge Base")

    if stats_data:
        st.markdown(f"📄 **Document**: `{stats_data.get('indexed_document', 'N/A')}`")
        st.markdown(f"📑 **Total Pages**: `{stats_data.get('total_pages', 0)} pages`")
        st.markdown(f"🧩 **Indexed Chunks**: `{stats_data.get('total_chunks', 0)} chunks`")
        st.markdown(f"📐 **Chunk Size**: `{stats_data.get('chunk_size', 900)} / {stats_data.get('chunk_overlap', 120)} overlap`")
        st.markdown(f"🧠 **Embedding**: `{stats_data.get('embedding_model', 'all-MiniLM-L6-v2')}` (384-dim)")
        st.markdown(f"⚡ **Vector DB**: `FAISS {stats_data.get('vector_database_status', 'active').upper()}`")
        st.caption(f"SHA-256: `{stats_data.get('pdf_sha256', '')[:16]}...`")
    else:
        st.warning("⚠️ Knowledge base statistics unavailable. Connect backend to inspect index.")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True, type="secondary"):
        clear_session_history(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.caption("Session ID: " + st.session_state.session_id[:8] + "...")


# ==========================================
# MAIN INTERFACE
# ==========================================

# Header Banner
st.markdown(
    """
    <div class="titan-header">
        <h1 class="titan-title">🛡️ TitanCloud SRE Incident Assistant</h1>
        <p class="titan-subtitle">
            Enterprise Site Reliability Engineering AI assistant grounded exclusively in the 
            official 50-page TitanCloud SRE Incident Runbook. Answers include verified original PDF page citations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Suggested Questions Section (Rendered when no active conversation or at user preference)
if len(st.session_state.messages) == 0:
    st.markdown("#### 💡 Quick Incident & Runbook Inquiries")
    st.caption("Select a sample question to query the verified 50-page enterprise runbook:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**⚡ Escalation & Timeline**")
        if st.button("P0 escalation timeline in PagerDuty", key="sq_p0", use_container_width=True):
            st.session_state.pending_question = "What is the escalation timeline when a Severity-1 (P0) alert triggers in PagerDuty?"
            st.rerun()
        if st.button("Who is notified at 15 minutes?", key="sq_p0_15", use_container_width=True):
            st.session_state.pending_question = "Under Section 9.1, who is paged if a Severity-1 alert remains unacknowledged for 15 minutes?"
            st.rerun()

    with col2:
        st.markdown("**📊 SLAs & Guarantees**")
        if st.button("Enterprise uptime SLA guarantee", key="sq_sla", use_container_width=True):
            st.session_state.pending_question = "What uptime guarantee SLA does TitanCloud provide to enterprise customers?"
            st.rerun()
        if st.button("SLA service credit schedule", key="sq_credit", use_container_width=True):
            st.session_state.pending_question = "What are the SLA service credit tiers applied when monthly uptime falls below thresholds?"
            st.rerun()

    with col3:
        st.markdown("**🔧 Error Codes & Diagnostics**")
        if st.button("Troubleshoot ERR_PG_REPLICA_LAG", key="sq_lag", use_container_width=True):
            st.session_state.pending_question = "How do you troubleshoot the error code ERR_PG_REPLICA_LAG in Patroni clusters?"
            st.rerun()
        if st.button("K8s control plane pod eviction", key="sq_k8s", use_container_width=True):
            st.session_state.pending_question = "What is the pod eviction threshold for Kubernetes distributed control planes?"
            st.rerun()

    st.markdown("---")


# Display Conversation History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🛡️"):
            st.markdown(msg["content"])

            # Render collapsible Source Citations
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📚 Verified Sources & Citations ({len(sources)} runbook excerpts)", expanded=False):
                    for idx, src in enumerate(sources, 1):
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <span class="citation-badge">Page {src.get('page_number', 'N/A')}</span>
                                <strong>{src.get('filename', 'titancloud_sre_incident_runbook_50p.pdf')}</strong>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.code(src.get("text", "").strip(), language="markdown")
            elif not msg.get("is_sufficient", True):
                st.caption("ℹ️ No authoritative excerpts matched the relevance threshold for this query.")


# Determine question input (from chat_input or suggested question pill)
prompt_input = st.chat_input("Ask a question about SRE procedures, error codes, SLAs, or on-call escalations...")
active_question = None

if prompt_input:
    active_question = prompt_input
elif st.session_state.pending_question:
    active_question = st.session_state.pending_question
    st.session_state.pending_question = None

# Process User Question
if active_question:
    # 1. Render and append user question
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(active_question)

    st.session_state.messages.append({"role": "user", "content": active_question})

    # 2. Query backend with meaningful loading spinner
    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("Searching the SRE runbook & generating grounded answer..."):
            response_data = send_chat_message(
                question=active_question,
                session_id=st.session_state.session_id,
                chat_history=st.session_state.messages[:-1],
            )

        answer_text = response_data.get("answer", "")
        sources_list = response_data.get("sources", [])
        is_sufficient = response_data.get("is_sufficient", True)

        st.markdown(answer_text)

        if sources_list:
            with st.expander(f"📚 Verified Sources & Citations ({len(sources_list)} runbook excerpts)", expanded=False):
                for idx, src in enumerate(sources_list, 1):
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <span class="citation-badge">Page {src.get('page_number', 'N/A')}</span>
                            <strong>{src.get('filename', 'titancloud_sre_incident_runbook_50p.pdf')}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.code(src.get("text", "").strip(), language="markdown")

        # Save assistant turn to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "sources": sources_list,
                "confidence": response_data.get("confidence"),
                "is_sufficient": is_sufficient,
            }
        )

    st.rerun()
