"""
Streamlit UI for the multi-agent research pipeline.

Drop this file into the root of `multi-agent-system/` (next to pipeline.py,
agents.py, tools.py) and run:

    streamlit run app.py

It reuses your existing agents/tools/pipeline logic directly (no changes
needed to pipeline.py) but drives it step-by-step so progress, sources,
report, and critic feedback all render live in the browser instead of
scrolling past in a terminal.
"""

from datetime import datetime

import streamlit as st

try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
except ImportError:  # keep the app usable even if this exact class isn't importable
    class ChatGoogleGenerativeAIError(Exception):
        pass

from pipeline import extract_urls
from agents import writer_chain, critic_chain
from tools import web_search, scrape_url


# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Research Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# STYLES  (high-contrast, colorful, dark theme)
# ------------------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --text-main: #f5f6ff;
        --text-dim:  #c7c9e0;
        --accent-blue:   #7fb0ff;
        --accent-green:  #4fe3a3;
        --accent-pink:   #ff8fc0;
        --accent-orange: #ffb86c;
        --accent-purple: #b78bff;
    }

    .stApp {
        background: radial-gradient(circle at top left, #1b1d2f 0%, #0b0c14 65%);
        color: var(--text-main);
    }

    /* Make ALL default streamlit text bright and readable */
    section.main, .stMarkdown, .stText, p, span, li, label, div[data-testid="stMarkdownContainer"] {
        color: var(--text-main) !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-dim) !important;
    }

    /* HERO BANNER */
    .hero {
        padding: 1.8rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #5b4bff 0%, #9b4bff 45%, #ff4b9e 100%);
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 34px rgba(120, 80, 255, 0.45);
    }
    .hero h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 2.1rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }
    .hero p {
        color: rgba(255,255,255,0.95) !important;
        margin: 0.35rem 0 0 0;
        font-size: 1.02rem;
        font-weight: 500;
    }

    /* TABS */
    button[data-baseweb="tab"] {
        color: var(--text-dim) !important;
        font-weight: 600;
        font-size: 0.98rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: var(--accent-pink) !important;
        height: 3px !important;
    }

    /* STATUS BOXES (st.status) */
    div[data-testid="stStatusWidget"], details {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stStatusWidget"] p, details summary p, details summary span {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* BADGES */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-blue   { background: rgba(90,140,255,0.22);  color: var(--accent-blue); }
    .badge-green  { background: rgba(60,220,150,0.22);  color: var(--accent-green); }
    .badge-pink   { background: rgba(255,90,170,0.22);  color: var(--accent-pink); }
    .badge-orange { background: rgba(255,160,60,0.22);  color: var(--accent-orange); }

    /* URL PILLS */
    .url-pill {
        display: block;
        padding: 0.6rem 1rem;
        margin-bottom: 0.45rem;
        border-radius: 10px;
        background: rgba(90,140,255,0.12);
        border-left: 4px solid var(--accent-blue);
        color: #dce8ff !important;
        font-size: 0.9rem;
        word-break: break-all;
        text-decoration: none;
        transition: background 0.15s ease;
    }
    .url-pill:hover {
        background: rgba(90,140,255,0.22);
    }

    /* REPORT / FEEDBACK CONTENT BOXES */
    .report-box, .report-box p, .report-box li, .report-box strong,
    .report-box h1, .report-box h2, .report-box h3, .report-box em {
        color: #f2f3ff !important;
    }
    .report-box {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        line-height: 1.65;
        font-size: 1rem;
    }
    .report-box h1, .report-box h2, .report-box h3 {
        color: #ffffff !important;
        border-bottom: 1px solid rgba(255,255,255,0.15);
        padding-bottom: 0.3rem;
    }
    .report-box code {
        background: rgba(255,255,255,0.08);
        color: var(--accent-green) !important;
        padding: 0.1rem 0.35rem;
        border-radius: 6px;
    }

    .feedback-box, .feedback-box p, .feedback-box li, .feedback-box strong,
    .feedback-box h1, .feedback-box h2, .feedback-box h3, .feedback-box em {
        color: #fff3e6 !important;
    }
    .feedback-box {
        background: rgba(255,180,80,0.09);
        border: 1px solid rgba(255,180,80,0.35);
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        line-height: 1.65;
        font-size: 1rem;
    }
    .feedback-box h1, .feedback-box h2, .feedback-box h3 {
        color: var(--accent-orange) !important;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14162a 0%, #0b0c14 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }
    section[data-testid="stSidebar"] button {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #ffffff !important;
    }

    /* PRIMARY BUTTON */
    button[kind="primary"] {
        background: linear-gradient(120deg, #6a5cff, #ff5cae) !important;
        border: none !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    /* TEXT INPUT */
    input[type="text"] {
        background: rgba(255,255,255,0.06) !important;
        color: #666666 !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
    }

    /* CODE BLOCKS / st.code, st.text */
    pre, code, .stCodeBlock, div[data-testid="stCode"] {
        color: #d6e3ff !important;
        background: rgba(255,255,255,0.05) !important;
    }
    div[data-testid="stText"] {
        color: var(--text-main) !important;
    }

    /* ALERT BOXES (info/warning/error) keep their own contrast, just brighten text */
    div[data-testid="stAlert"] p {
        color: #ffffff !important;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []   # list of {topic, report, feedback, urls, timestamp}
if "result" not in st.session_state:
    st.session_state.result = None


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🧠 Research Agent")
    st.caption("Multi-agent research pipeline")

    st.divider()
    st.markdown("**Pipeline steps**")
    st.markdown(
        "1. 🔎 Web search\n"
        "2. 🔗 Extract URLs\n"
        "3. 🕸️ Scrape sources\n"
        "4. ✍️ Draft report\n"
        "5. 🧐 Critic review"
    )

    st.divider()
    st.markdown("**Past runs**")
    if st.session_state.history:
        for i, run in enumerate(reversed(st.session_state.history)):
            if st.button(f"📄 {run['topic'][:28]}", key=f"hist_{i}"):
                st.session_state.result = run
    else:
        st.caption("No runs yet.")


# ------------------------------------------------------------------
# HERO / INPUT
# ------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🧠 Multi-Agent Research System</h1>
        <p>Search → Scrape → Write → Critique — powered by your agents.py / tools.py</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Impact of quantum computing on cryptography",
        label_visibility="collapsed",
    )
with col2:
    run_clicked = st.button("🚀 Run Research", use_container_width=True, type="primary")


# ------------------------------------------------------------------
# PIPELINE RUN (mirrors pipeline.run_research_pipeline, but streams to UI)
# ------------------------------------------------------------------

def run_pipeline_ui(topic: str):
    result = {"topic": topic, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}

    progress = st.progress(0, text="Starting up...")

    # STEP 1: SEARCH -------------------------------------------------
    with st.status("🔎 Step 1 — Searching the web...", expanded=True) as status:
        search_results = web_search.invoke(topic)
        result["search_results"] = search_results
        st.code(search_results[:2000] + ("..." if len(search_results) > 2000 else ""), language="text")
        status.update(label="✅ Web search complete", state="complete")
    progress.progress(15, text="Search complete")

    # STEP 2: EXTRACT URLS -------------------------------------------
    urls = extract_urls(search_results)
    result["urls"] = urls

    with st.status(f"🔗 Step 2 — Found {len(urls)} source URLs", expanded=True) as status:
        if not urls:
            st.error("No URLs found in search results. Stopping here.")
            status.update(label="⚠️ No URLs found", state="error")
            progress.progress(100, text="Done (no sources)")
            return result
        for u in urls:
            st.markdown(f'<a class="url-pill" href="{u}" target="_blank">{u}</a>', unsafe_allow_html=True)
        status.update(label=f"✅ {len(urls)} URLs extracted", state="complete")
    progress.progress(30, text="URLs extracted")

    # STEP 3: SCRAPE ---------------------------------------------------
    scraped_text = ""
    with st.status("🕸️ Step 3 — Scraping websites...", expanded=True) as status:
        scrape_progress = st.progress(0)
        for i, url in enumerate(urls, start=1):
            st.write(f"Scraping ({i}/{len(urls)}): `{url}`")
            page = scrape_url.invoke(url)
            scraped_text += f"\n\n=========================================================\nSOURCE URL:\n{url}\n\nCONTENT:\n{page}\n\n"
            scrape_progress.progress(i / len(urls))
        result["scraped_content"] = scraped_text
        status.update(label="✅ Scraping complete", state="complete")
    progress.progress(60, text="Sources scraped")

    # STEP 4: WRITER -----------------------------------------------------
    with st.status("✍️ Step 4 — Writing research report...", expanded=True) as status:
        research = f"\n\nSEARCH RESULTS\n\n{search_results}\n\n\nSCRAPED CONTENT\n\n{scraped_text}\n\n"
        report = writer_chain.invoke({"topic": topic, "research": research})
        result["report"] = report
        status.update(label="✅ Report drafted", state="complete")
    progress.progress(85, text="Report drafted")

    # STEP 5: CRITIC -----------------------------------------------------
    with st.status("🧐 Step 5 — Reviewing report...", expanded=True) as status:
        feedback = critic_chain.invoke({"report": report})
        result["feedback"] = feedback
        status.update(label="✅ Review complete", state="complete")
    progress.progress(100, text="Done!")

    return result


if run_clicked:
    if not topic.strip():
        st.warning("Enter a research topic first.")
    else:
        try:
            result = run_pipeline_ui(topic.strip())
            st.session_state.result = result
            st.session_state.history.append(result)
            st.balloons()
        except ChatGoogleGenerativeAIError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                st.error(
                    "🚫 Gemini API daily quota exceeded (free tier limit reached). "
                    "Wait for the quota reset, switch models, or enable billing."
                )
            else:
                st.error(f"Model error: {e}")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")


# ------------------------------------------------------------------
# RESULTS DISPLAY
# ------------------------------------------------------------------

result = st.session_state.result

if result:
    st.divider()
    st.markdown(f"## 📋 Results for: *{result['topic']}*")
    if "timestamp" in result:
        st.caption(f"Run at {result['timestamp']}")

    tab_report, tab_feedback, tab_sources, tab_raw = st.tabs(
        ["📝 Report", "🧐 Critic Feedback", "🔗 Sources", "🔍 Raw Search"]
    )

    with tab_report:
        if result.get("report"):
            st.markdown(f'<div class="report-box">{result["report"]}</div>', unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download report (.md)",
                data=result["report"],
                file_name=f"{result['topic'][:40].replace(' ', '_')}_report.md",
                mime="text/markdown",
            )
        else:
            st.info("No report generated for this run.")

    with tab_feedback:
        if result.get("feedback"):
            st.markdown(f'<div class="feedback-box">{result["feedback"]}</div>', unsafe_allow_html=True)
        else:
            st.info("No feedback available.")

    with tab_sources:
        urls = result.get("urls", [])
        if urls:
            st.markdown(
                f'<span class="badge badge-blue">{len(urls)} sources</span>',
                unsafe_allow_html=True,
            )
            for u in urls:
                st.markdown(f'<a class="url-pill" href="{u}" target="_blank">{u}</a>', unsafe_allow_html=True)
        else:
            st.info("No sources for this run.")

    with tab_raw:
        st.text(result.get("search_results", "No raw search data."))

else:
    st.info("👋 Enter a topic above and click **Run Research** to start the pipeline.")