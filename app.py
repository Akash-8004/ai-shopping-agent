import os
import sys
import tempfile
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

sys.path.insert(0, os.path.dirname(__file__))

from shopping_agent import create_shopping_agent, is_shopping_request  # noqa: E402

# ─── Page configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="ShopSmart AI – Your Personal Shopping Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS injection ───────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ─── Google Fonts ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ─── Root variables ─── */
:root {
    --bg-primary: #0a0f1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.7);
    --glass: rgba(255, 255, 255, 0.04);
    --glass-border: rgba(255, 255, 255, 0.08);
    --accent-emerald: #10b981;
    --accent-purple: #8b5cf6;
    --accent-blue: #3b82f6;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --gradient-1: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
    --gradient-2: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
    --shadow-glow: 0 0 30px rgba(16, 185, 129, 0.1);
}

/* ─── Global resets ─── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 50%, #0f172a 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ─── Hide Streamlit branding ─── */
#MainMenu, footer, header {visibility: hidden;}

/* ─── App header ─── */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1rem 1rem;
    margin-bottom: 1rem;
}

.app-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}

.app-header p {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 300;
}

/* ─── Chat messages ─── */
[data-testid="stChatMessage"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    animation: fadeSlideIn 0.4s ease-out !important;
}

/* user message accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid var(--accent-purple) !important;
    background: rgba(139, 92, 246, 0.04) !important;
}

/* assistant message accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid var(--accent-emerald) !important;
    background: rgba(16, 185, 129, 0.04) !important;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ─── Chat input ─── */
[data-testid="stChatInput"] {
    background: transparent !important;
}

[data-testid="stChatInput"] textarea {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.9rem 1.2rem !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-emerald) !important;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.15) !important;
}

[data-testid="stChatInput"] button {
    background: var(--gradient-1) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stChatInput"] button:hover {
    transform: scale(1.06) !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
}

/* ─── Sidebar styling ─── */
.sidebar-header {
    text-align: center;
    padding: 1.5rem 0.5rem 1rem 0.5rem;
}

.sidebar-header h2 {
    font-size: 1.4rem;
    font-weight: 700;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.sidebar-header p {
    color: var(--text-muted);
    font-size: 0.82rem;
}

.stat-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    backdrop-filter: blur(10px);
}

.stat-card .label {
    color: var(--text-muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.15rem;
}

.stat-card .value {
    color: var(--text-primary);
    font-size: 1rem;
    font-weight: 600;
}

/* ─── Welcome card ─── */
.welcome-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    max-width: 640px;
    margin: 2rem auto;
    box-shadow: var(--shadow-glow);
}

.welcome-card h2 {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.welcome-card p {
    color: var(--text-secondary);
    font-size: 0.92rem;
    margin-bottom: 1.5rem;
    line-height: 1.5;
}

.welcome-features {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    justify-content: center;
    margin-bottom: 1.5rem;
}

.feature-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 100px;
    padding: 0.4rem 0.9rem;
    font-size: 0.8rem;
    color: var(--accent-emerald);
    font-weight: 500;
}

/* ─── Tool activity badge ─── */
.tool-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 8px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    color: var(--accent-blue);
    font-weight: 500;
    margin-right: 0.4rem;
    margin-bottom: 0.3rem;
}

.tools-used-row {
    display: flex;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}

/* ─── Buttons ─── */
.stButton > button {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.25s ease !important;
    padding: 0.5rem 1.2rem !important;
}

.stButton > button:hover {
    background: rgba(16, 185, 129, 0.08) !important;
    border-color: var(--accent-emerald) !important;
    color: var(--accent-emerald) !important;
    transform: translateY(-1px) !important;
}

/* ─── File uploader ─── */
[data-testid="stFileUploader"] {
    background: var(--glass) !important;
    border: 1px dashed var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-emerald) !important;
}

/* ─── Expander ─── */
[data-testid="stExpander"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
}

/* ─── Divider ─── */
hr {
    border-color: var(--glass-border) !important;
    opacity: 0.5;
}

/* ─── Spinner ─── */
.stSpinner > div {
    border-top-color: var(--accent-emerald) !important;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255,255,255,0.2);
}

/* ─── Toast / alerts ─── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}

/* ─── Guardrail badge ─── */
.guardrail-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: 8px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    color: #f59e0b;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─── Tool name mapping for display ───────────────────────────────────────────

TOOL_DISPLAY_NAMES = {
    "search_products": "🔍 Product Search",
    "get_rating": "⭐ Rating Lookup",
    "analyze_product_image": "📷 Image Analysis",
    "checkout_for_visitor": "🛒 Checkout",
    "save_preference_for_visitor": "💾 Save Preference",
    "get_preferences_for_visitor": "📋 Load Preferences",
    "get_order_history_for_visitor": "📦 Order History",
}


# ─── Helper: extract tool calls from agent response ──────────────────────────

def extract_tool_names(messages: list) -> list[str]:
    """Pull unique tool names from tool-call messages in the response."""
    tools_used = []
    seen = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and hasattr(msg, "name"):
            name = msg.name
            if name not in seen:
                seen.add(name)
                tools_used.append(name)
        elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    tools_used.append(name)
    return tools_used


# ─── Session state initialisation ────────────────────────────────────────────

def init_session_state() -> None:
    if "visitor_id" not in st.session_state:
        st.session_state.visitor_id = str(uuid.uuid4())

    if "agent" not in st.session_state:
        st.session_state.agent = create_shopping_agent(st.session_state.visitor_id)
        st.session_state.config = {
            "configurable": {
                "thread_id": st.session_state.visitor_id,
            }
        }

    if "history" not in st.session_state:
        # Each entry: {"role": str, "content": str, "tools": list[str]}
        st.session_state.history = []

    if "image_path" not in st.session_state:
        st.session_state.image_path = None


init_session_state()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <h2>🛒 ShopSmart AI</h2>
            <p>Intelligent Shopping Assistant</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Session stats ──
    msg_count = len(st.session_state.history)
    user_msgs = sum(1 for m in st.session_state.history if m["role"] == "user")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""<div class="stat-card">
                <div class="label">Messages</div>
                <div class="value">{msg_count}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="stat-card">
                <div class="label">Session</div>
                <div class="value">{st.session_state.visitor_id[:8]}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Image upload ──
    st.markdown("##### 📷 Image Search")
    uploaded = st.file_uploader(
        "Upload a product photo to identify & search it",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)
        suffix = os.path.splitext(uploaded.name)[-1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        tmp.close()
        st.session_state.image_path = tmp.name
        st.success("✅ Image ready — send a message to search!")
    else:
        st.session_state.image_path = None

    st.markdown("---")

    # ── Sample prompts ──
    st.markdown("##### 💡 Try These")
    sample_prompts = [
        "Show me organic honey under $20",
        "What olive oils do you have?",
        "Show me top-rated almonds",
        "Remember I prefer organic products",
        "What have I ordered before?",
    ]
    for prompt in sample_prompts:
        if st.button(prompt, key=f"sample_{prompt}", use_container_width=True):
            st.session_state["_pending_prompt"] = prompt
            st.rerun()

    st.markdown("---")

    # ── Clear chat ──
    if st.button("🗑️  Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.agent = create_shopping_agent(st.session_state.visitor_id)
        st.session_state.config = {
            "configurable": {
                "thread_id": st.session_state.visitor_id,
            }
        }
        st.rerun()

    st.markdown(
        """
        <div style="text-align:center; padding-top:1rem;">
            <p style="color: var(--text-muted); font-size: 0.72rem;">
                Powered by Groq · LangGraph · LangChain
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Main chat area ─────────────────────────────────────────────────────────

# App header
st.markdown(
    """
    <div class="app-header">
        <h1>🛒 ShopSmart AI</h1>
        <p>Your intelligent personal shopping assistant — search products, compare ratings, and order with ease</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Welcome card (shown when chat is empty) ──
if not st.session_state.history:
    st.markdown(
        """
        <div class="welcome-card">
            <h2>👋 Welcome to ShopSmart AI!</h2>
            <p>
                I can help you discover products, check ratings & reviews,
                remember your preferences, and place orders — all through conversation.
            </p>
            <div class="welcome-features">
                <span class="feature-chip">🔍 Product Search</span>
                <span class="feature-chip">⭐ Ratings & Reviews</span>
                <span class="feature-chip">📷 Image Search</span>
                <span class="feature-chip">🛒 Easy Checkout</span>
                <span class="feature-chip">💾 Preferences</span>
                <span class="feature-chip">📦 Order History</span>
            </div>
            <p style="font-size:0.82rem; color: var(--text-muted);">
                Try asking: <em>"Show me organic honey under $20"</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Render chat history ──
for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        # Show tool badges for assistant messages
        if entry["role"] == "assistant" and entry.get("tools"):
            badges_html = '<div class="tools-used-row">'
            for t in entry["tools"]:
                display = TOOL_DISPLAY_NAMES.get(t, f"🔧 {t}")
                badges_html += f'<span class="tool-badge">{display}</span>'
            badges_html += "</div>"
            st.markdown(badges_html, unsafe_allow_html=True)
        st.markdown(entry["content"])


# ── Handle input (from chat box or sample prompt button) ──
pending_prompt = st.session_state.pop("_pending_prompt", None)
user_input = st.chat_input(
    "Ask me to find products, check ratings, or place an order…"
)

active_input = pending_prompt or user_input

if active_input:
    # Attach image if present
    if st.session_state.image_path:
        message_text = f"Image path: {st.session_state.image_path}\n\n{active_input}"
        display_text = f"📷 *[image attached]*\n\n{active_input}"
        image_tmp_path = st.session_state.image_path
        st.session_state.image_path = None
    else:
        message_text = active_input
        display_text = active_input
        image_tmp_path = None

    # Append & render user message
    st.session_state.history.append(
        {"role": "user", "content": display_text, "tools": []}
    )
    with st.chat_message("user"):
        st.markdown(display_text)

    # ── Guardrail check ──
    is_on_topic = True
    if not st.session_state.image_path:  # skip guardrail for image searches
        try:
            is_on_topic = is_shopping_request(active_input)
        except Exception:
            is_on_topic = True  # fail-open: let the agent handle it

    # Invoke agent (or block with guardrail)
    with st.chat_message("assistant"):
        if not is_on_topic:
            st.markdown(
                '<span class="guardrail-badge">🛡️ Off-topic detected</span>',
                unsafe_allow_html=True,
            )
            reply = (
                "I'm a **shopping assistant** 🛒 — I can help you with:\n\n"
                "- 🔍 Searching for products\n"
                "- ⭐ Checking ratings & reviews\n"
                "- 🛒 Placing orders\n"
                "- 💾 Saving your preferences\n"
                "- 📦 Viewing order history\n\n"
                "Please ask me something shopping-related!"
            )
            tools_used = []
        else:
            with st.spinner("🧠 Thinking…"):
                try:
                    response = st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content=message_text)]},
                        config=st.session_state.config,
                    )
                    reply = response["messages"][-1].content
                    tools_used = extract_tool_names(response["messages"])
                except Exception as e:
                    err_str = str(e).lower()
                    if "rate_limit" in err_str or "429" in err_str:
                        reply = (
                            "⏳ **Rate limit reached** — the AI provider is temporarily "
                            "throttling requests.\n\n"
                            "Please wait **a few seconds** and send your message again. "
                            "The system will auto-retry up to 3 times, but the limit may "
                            "still be exceeded on the free tier."
                        )
                    else:
                        reply = (
                            f"⚠️ Sorry, something went wrong while processing your request.\n\n"
                            f"**Error:** `{type(e).__name__}: {e}`\n\n"
                            f"Please try again or rephrase your question."
                        )
                    tools_used = []

        # Show tool badges
        if tools_used:
            badges_html = '<div class="tools-used-row">'
            for t in tools_used:
                display = TOOL_DISPLAY_NAMES.get(t, f"🔧 {t}")
                badges_html += f'<span class="tool-badge">{display}</span>'
            badges_html += "</div>"
            st.markdown(badges_html, unsafe_allow_html=True)

        st.markdown(reply)

    # Persist
    st.session_state.history.append(
        {"role": "assistant", "content": reply, "tools": tools_used}
    )

    # Clean up temp image file
    if image_tmp_path:
        try:
            os.unlink(image_tmp_path)
        except OSError:
            pass

    st.rerun()
