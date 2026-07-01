# Simulate offline functionality
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Start:  streamlit run 05_dashboard.py

import os
import json
import sqlite3
import collections
import datetime as dt
from email.utils import parsedate_to_datetime

import numpy as np
import streamlit as st
import plotly.express as px
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import ollama

# -- LangChain/LangGraph components for the agent (Section 9)
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from agent_core import build_agent            # LangGraph-Engine (build_agent)

COMPANY = "Lufthansa"
INDUSTRY = "Luftfahrt / Aviation"
LLM_MODEL = "qwen3:8b"
SENTIMENT_COLORS = {"positive": "#2a722a",
                    "neutral": "#979da2",
                    "negative": "#a02929"}
MIN_EVIDENCE_LENGTH = 80
# Prompt treatment to combat hallucinations.
GUARD = ("Rely exclusively on the evidence below and do not invent any facts or figures. "
        "You MAY draw reasonable strategic conclusions from the evidence. Write "
        "'INSUFFICIENT EVIDENCE' ONLY if the evidence provides nothing relevant to the question.")

st.set_page_config(
    page_title="Lufthansa Intelligence Dashboard", layout="wide")


# Find database
def find_db():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "vectors.db",
        os.path.join(here, "vectors.db"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "vectors.db"


# Load data and models
@st.cache_resource
def load_all():
    db = sqlite3.connect(find_db())
    # Determine the column name robustly: the DB may use "vector" or "vektor"
    cols = [r[1] for r in db.execute("PRAGMA table_info(docs)")]
    vcol = "vector" if "vector" in cols else "vektor"
    lines = db.execute(
        f"SELECT title, text, source, date, url, {vcol} FROM docs").fetchall()

    title, texts, sources, dates, urls = [], [], [], [], []
    for z in lines:
        title.append(z[0])
        texts.append(z[1])
        sources.append(z[2])
        dates.append(z[3])
        urls.append(z[4])
    vectors = np.array([json.loads(z[5]) for z in lines])   # z[5] corresponds to the column vector

    model = SentenceTransformer("intfloat/multilingual-e5-base")
    mood = pipeline("sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment")
    return title, texts, sources, dates, urls, vectors, model, mood


title, texts, sources, dates, urls, vectors, model, mood = load_all()


# Helper - Date as a sortable number
def date_key(s):
    if not s:
        return 0.0
    try:                                   # RSS format, e.g. "Fri, 05 Jul 2024 10:56:38 GMT"
        return parsedate_to_datetime(s).timestamp()
    except Exception:
        pass
    try:                                   # ISO format, e.g. "2024-07-05T10:56:38+00:00"
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0

# RAG: returns the k most similar documents (title, text, source, URL, score).
def search(question, k=5):
    f = model.encode("query: " + question, normalize_embeddings=True)
    similarity = vectors @ f
    hits = []
    for i in similarity.argsort()[::-1]:
        # Title-only
        if len(texts[i].strip()) < MIN_EVIDENCE_LENGTH:   # skip title-only entries
            continue
        hits.append((title[i], texts[i], sources[i],
                    urls[i], float(similarity[i])))
        if len(hits) >= k:
            break
    return hits

# Format the hits as numbered evidence for the prompt.
def format_evidence(hits):
    evidence = ""
    number = 1
    for t, text, source, url, score in hits:
        evidence = evidence + "Evidence " + str(number) + ": " + text[:300] \
            + " (Source: " + source + " - " + url + ")\n"
        number = number + 1
    return evidence

# Streaming for immediate text output
def stream_llm(prompt):
    try:
        for chunk in ollama.chat(model=LLM_MODEL,
                                 messages=[
                                     {"role": "user", "content": prompt}],
                                 stream=True):
            yield chunk["message"]["content"]
    except Exception as e:
        yield ("LLM unavailable, service active? (Terminal: `ollama serve`, "
               f"Modell: `ollama pull {LLM_MODEL}`)\n\nDetails: {e}")


@st.cache_data  # Streamlit decorator (caches the return value)
def sentiment_per_doc():
    def label(text):
        stars = int(mood(text[:512])[0]["label"][0])   # "4 stars" -> 4
        if stars <= 2:
            return "negative"
        if stars == 3:
            return "neutral"
        return "positive"
    out = []
    for i in range(len(texts)):
        if texts[i].strip():
            # (timestamp, label)
            out.append((date_key(dates[i]), label(texts[i])))
    return out


# label count
def sentiment_counts():
    return collections.Counter(lbl for ts, lbl in sentiment_per_doc())


# Evidence docs
def show_evidence(hits):
    with st.expander("Documents used"):
        for t, text, source, url, score in hits:
            st.markdown(
                f"- **{t}** — {source} · Score {round(score, 2)}  \n  {url}")


# -- LangChain agent stack (reuses the already loaded e5 and nlptown models)
class E5QueryEmbeddings(Embeddings):
    """multilingual-e5-base (schon geladen) als LangChain-Embeddings verpackt."""

    def __init__(self, st_model):
        self.model = st_model

    def embed_documents(self, texts_):
        return self.model.encode(["passage: " + t for t in texts_],
                                 normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode("query: " + text,
                                 normalize_embeddings=True).tolist()


@st.cache_resource
def build_vectorstore():
    """FAISS using the e5 vectors already stored in the database (no re-embedding)."""
    pairs, metas = [], []
    for i in range(len(texts)):
        if len(texts[i].strip()) < MIN_EVIDENCE_LENGTH:
            continue
        pairs.append((texts[i], vectors[i].tolist()))
        metas.append({"title": title[i], "source": sources[i], "url": urls[i]})
    return FAISS.from_embeddings(pairs, E5QueryEmbeddings(model), metadatas=metas)


@st.cache_resource
def get_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0)


@tool
def search_evidence(query: str, k: int = 5) -> list:
    """Semantic search in the Lufthansa knowledge base; returns relevant documents."""
    out = []
    for doc, dist in build_vectorstore().similarity_search_with_score(query, k=k):
        cos = round(1.0 - dist / 2.0, 3)               # Einheitsvektoren: cos = 1 - L2^2/2
        m = doc.metadata
        out.append({"title": m.get("title", ""), "text": doc.page_content,
                    "source": m.get("source", ""), "url": m.get("url", ""), "score": cos})
    return out


def agent_search(query, k=5):
    """Adapter: routes the agent search via the LangChain-FAISS-@tool."""
    docs = search_evidence.invoke({"query": query, "k": k})
    return [(d["title"], d["text"], d["source"], d["url"], d["score"]) for d in docs]


def sentiment_of_texts(text_list):
    """nlptown 1–5 stars -> negative / neutral / positive distribution."""
    def label(text):
        stars = int(mood(text[:512])[0]["label"][0])
        if stars <= 2:
            return "negative"
        if stars == 3:
            return "neutral"
        return "positive"
    return dict(collections.Counter(
        label(t) for t in text_list if t and t.strip()))


@tool
def analyze_sentiment(query: str, k: int = 8):
    """Retrieve documents for a topic and measure public sentiment."""
    hits = search_evidence.invoke({"query": query, "k": k})
    dist = sentiment_of_texts([h["text"] for h in hits])
    # beides zurueckgeben: Sentiment UND Dokumente (bleiben als Evidenz erhalten)
    return {"sentiment": dist, "hits": hits}


def analyze(query, k=8):
    """Adapter: runs the analyse_sentiment tool and returns (dist, Tupel)."""
    result = analyze_sentiment.invoke({"query": query, "k": k})
    dist = result["sentiment"]
    docs = result["hits"]
    hits = [(d["title"], d["text"], d["source"], d["url"], d["score"]) for d in docs]
    return dist, hits


# DASHBOARD
st.title("Lufthansa – Strategic Intelligence Dashboard")
st.caption("A Strategic Intelligence Agent for Lufthansa")
st.caption("Note: Sections 3, 4, 5, 7 and 8 generate their content using an LLM (Ollama/qwen3:8b). "
           "The response is returned in batches; the very first request takes longer because the model "
           "is loaded into memory.")

# Section 1: Company Overview
st.header("1. Company Overview")
parsed = [date_key(d) for d in dates if date_key(d) > 0]
last_update = (dt.datetime.fromtimestamp(max(parsed)).strftime("%d.%m.%Y")
               if parsed else "—")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Company", COMPANY)
c2.metric("Industry", INDUSTRY)
c3.metric("Documents", len(title))
c4.metric("Data sources", len(set(sources)))
c5.metric("Last updated", last_update)

# Section 2: Market Intelligence
st.header("2. Market intelligence")
st.caption(
    "Latest news, competitor updates and industry news (most recent first)")
quelle = st.selectbox("Filter by source", ["(all)"] + sorted(set(sources)))
order = sorted(range(len(title)),
               key=lambda i: date_key(dates[i]), reverse=True)
shown = 0
for i in order:
    if quelle != "(all)" and sources[i] != quelle:
        continue
    st.markdown(f"**[{title[i]}]({urls[i]})**  \n*{sources[i]} · {dates[i]}*")
    shown += 1
    if shown >= 12:
        break

# Section 3: Opportunities Monitor
st.header("3. Opportunity Monitor")
st.caption("Opportunities identified by the AI agent, with impact, evidence and confidence")
if st.button("Analyze opportunities"):
    hits = search(
        "opportunities, new markets, partnerships, growth, new technologies for Lufthansa", k=6)
    prompt = (
        "You are a strategy adviser to the Lufthansa Executive Board. Analyze ONLY the following "
        "evidence and name the 3 most important OPPORTUNITIES. For each opportunity give exactly these fields:\n"
        "TITLE | IMPACT (High/Medium/Low) | EVIDENCE (No.) | CONFIDENCE (High/Medium/Low)\n\n"
        + GUARD + "\n\n"
        + format_evidence(hits))
    with st.spinner(f"Generating response with {LLM_MODEL} … (The initial call may take a moment)"):
        st.write_stream(stream_llm(prompt))
    show_evidence(hits)

# Section 4: Risk Monitor
st.header("4. Risk Monitor")
st.caption(
    "Risks identified by the AI agent, with category, severity, evidence and confidence")
if st.button("Analyze risks"):
    hits = search(
        "risks, competition, regulation, strike, negative sentiment, supply chain at Lufthansa", k=6)
    prompt = (
        "You are a strategy adviser to the Lufthansa Executive Board. Analyze ONLY the following "
        "evidence and name the 3 biggest RISKS. For each risk give exactly these fields:\n"
        "TITLE | CATEGORY (Competition/Regulation/Sentiment/Supply chain/…) | "
        "SEVERITY (High/Medium/Low) | EVIDENCE (No.) | CONFIDENCE (High/Medium/Low)\n\n"
        + GUARD + "\n\n"
        + format_evidence(hits))
    with st.spinner(f"Generating response with {LLM_MODEL} … (The initial call may take a moment)"):
        st.write_stream(stream_llm(prompt))
    show_evidence(hits)

# Section 5: Trend Monitor
st.header("5. Trend Monitor")
st.caption(
    "Technology and industry trends identified by the AI agent, with type, evidence and confidence")

# Trend chart: shows the volume of news per week (is interest rising or falling?)
TREND_WEEKS = 12                                  
trend_dates = sorted(date_key(d) for d in dates if date_key(d) > 0)
if trend_dates:
    window = trend_dates[-1] - TREND_WEEKS * 7 * 86400   # relative to the most recent document
    buckets = collections.Counter()
    for ts in trend_dates:
        if ts < window:
            continue
        day = dt.datetime.fromtimestamp(ts)
        week_start = day - dt.timedelta(days=day.weekday())   # Monday of that week
        buckets[week_start.strftime("%Y-%m-%d")] += 1
    weeks = sorted(buckets)
    history = px.line(
        {"Week": weeks, "Documents": [buckets[w] for w in weeks]},
        x="Week", y="Documents", markers=True,
        title=f"Coverage volume over time (documents per week, last {TREND_WEEKS} weeks)",
    )
    history.update_layout(xaxis_title="", yaxis_title="Documents")
    # Druckmoeglichkeit: Plotly-Toolbar fest einblenden; Kamera-Icon laedt das Diagramm als PNG
    chart_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "lufthansa_trend_monitor",
            "scale": 2,                                # hoehere Aufloesung fuer den Druck
        },
    }
    st.plotly_chart(history, use_container_width=True, config=chart_config)
    st.caption(
        "Number of indexed documents per week – shows whether the topic is gaining or losing momentum. "
        "Use the 📷 icon (top-right of the chart) to download it as a PNG for printing.")
else:
    st.info("No time series available (no parsable dates).")

if st.button("Analyze trends"):
    hits = search(
        "trends, new technologies, SAF sustainable aviation fuel, digitalization, "
        "consolidation, customer behaviour, industry developments in aviation", k=6)
    prompt = (
        "You are a strategy adviser to the Lufthansa Executive Board. Analyze ONLY the following "
        "evidence and name the 3 most important emerging TRENDS/TECHNOLOGIES. For each trend give exactly these fields:\n"
        "TITLE | TYPE (Technology/Customer behaviour/Industry) | EVIDENCE (No.) | CONFIDENCE (High/Medium/Low)\n\n"
        + GUARD + "\n\n"
        + format_evidence(hits))
    with st.spinner(f"Generating response with {LLM_MODEL} … (The initial call may take a moment)"):
        st.write_stream(stream_llm(prompt))
    show_evidence(hits)

# Section 6: Sentiment Analysis
st.header("6. Sentiment Analysis")
data = sentiment_per_doc()
counter = collections.Counter(lbl for ts, lbl in data)
order = [s for s in ["positive", "neutral", "negative"] if s in counter]

col_pie, col_trend = st.columns(2)

# Left: current overall distribution (all documents)
with col_pie:
    fig = px.pie(
        names=order,
        values=[counter[s] for s in order],
        color=order,
        color_discrete_map=SENTIMENT_COLORS,
        title="Public sentiment (all documents)",
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(legend_title_text="Sentiment")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Based on: {sum(counter.values())} evaluated documents")

# Right: sentiment trend over time (last 30 days, stacked daily)
with col_trend:
    dated = [(ts, lbl) for ts, lbl in data if ts > 0]
    if dated:
        newest = max(ts for ts, lbl in dated)
        window = newest - 30 * 86400          # 30 days before the most recent document
        recent = [(dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d"), lbl)
                  for ts, lbl in dated if ts >= window]
        days = sorted(set(tag for tag, lbl in recent))
        x, y, c = [], [], []
        for tag in days:
            for s in ["positive", "neutral", "negative"]:
                x.append(tag)
                c.append(s)
                y.append(sum(1 for d2, l2 in recent if d2 == tag and l2 == s))
        trend = px.bar(
            {"Day": x, "Count": y, "Sentiment": c},
            x="Day", y="Count", color="Sentiment", barmode="stack",
            color_discrete_map=SENTIMENT_COLORS,
            title="Sentiment trend (last 30 days)",
        )
        trend.update_layout(legend_title_text="Sentiment",
                            xaxis_title="", yaxis_title="Documents")
        st.plotly_chart(trend, use_container_width=True)
        st.caption(
            f"{len(recent)} documents in the time window – shows whether sentiment is shifting")
    else:
        st.info("No time series available (no parsable dates).")

# Section 7: Strategic Recommendation
st.header("7. Strategic Recommendation")
st.caption("Evidence-based recommendation from the AI CEO agent (Task 6)")
question = st.text_input("Question for the AI CEO agent",
                         "What is the biggest risk for Lufthansa?")
if st.button("Generate recommendation"):
    hits = search(question, k=5)
    prompt = (
        "You are a strategy adviser to the Lufthansa Executive Board. Answer the question using "
        "ONLY the following evidence. Do not invent anything.\n"
        + GUARD + "\n\n"
        + f"Question: {question}\n\n"
        + format_evidence(hits) +
        "\nReply using exactly this structure:\n"
        "RECOMMENDATION: <a specific action>\n"
        "PRIORITY: <High/Medium/Low>\n"
        "SUPPORTING DOCUMENTS: <which evidence numbers you rely on>\n"
        "EXPECTED IMPACT: <benefit for the company>\n"
        "RISK: <financial/operational/strategic risk>")
    with st.spinner(f"Generating response with {LLM_MODEL} … (The initial call may take a moment)"):
        st.write_stream(stream_llm(prompt))
    show_evidence(hits)

# Section 8: CEO Briefing
st.header("8. CEO Briefing")
st.caption("Executive summary: What happened? Why does it matter? What to do?")
if st.button("Generate CEO briefing"):
    hits = search(
        "most important current developments, risks and opportunities for Lufthansa", k=8)
    sentiment = dict(sentiment_counts())
    prompt = (
        "You are the chief of staff to the Lufthansa Executive Board, writing a short CEO briefing. "
        f"The measured public sentiment across all documents is: {sentiment}.\n"
        + GUARD + "\n\n"
        + format_evidence(hits) +
        "\nReply concisely using exactly this structure:\n"
        "WHAT HAPPENED: <2-3 sentences>\n"
        "WHY IT MATTERS: <2-3 sentences>\n"
        "WHAT MANAGEMENT SHOULD DO: <2-3 concrete points>")
    with st.spinner(f"Generating response with {LLM_MODEL} … (The initial call may take a moment)"):
        st.write_stream(stream_llm(prompt))
    show_evidence(hits)

# Section 9: Autonomous Strategic Agent (LangChain / LangGraph)
st.header("9. Autonomous Strategic Agent (LangChain / LangGraph)")
st.caption("Built with LangGraph (StateGraph), ChatOllama, FAISS and LangChain @tool tools. "
           "Unlike the sections above (single prompt -> LLM+RAG -> answer), this agent works "
           "through the full workflow: Goal → Plan → Retrieve → Analyze → Decide → Recommend → "
           "Validate. It plans its own steps, chooses tools, gathers evidence, and validates its "
           "recommendation (with one automatic revision pass) before showing it.")
goal = st.text_input(
    "Strategic goal for the agent",
    "What is currently the biggest strategic risk for Lufthansa, and what should the board do about it?",
    key="agent_goal")
if st.button("Run autonomous agent"):
    agent = build_agent(get_llm(), agent_search, sentiment_of_texts, analyze)
    labels = {
        "plan": "Step 1: **Plan** – decomposing the goal into research steps …",
        "execute": "Step 2: **Retrieve + Analyze** – running tools (FAISS search & sentiment) …",
        "decide": "Step 3: **Decide + Recommend** – synthesising the evidence …",
        "validate": "Step 4: **Validate** – checking the draft against the evidence …",
    }
    final = {}
    decide_seen = False
    with st.status("Agent working …", expanded=True) as status:
        for chunk in agent.stream({"goal": goal}):
            for node, update in chunk.items():
                if node == "decide" and decide_seen:
                    st.write("Step rerun-execution: Validation flagged issues – running one revision pass …")
                if node in labels:
                    st.write(labels[node])
                if node == "decide":
                    decide_seen = True
                if isinstance(update, dict):
                    final.update(update)
        status.update(label="Agent finished", state="complete", expanded=False)

    # Plan
    st.subheader("Plan")
    plan = final.get("plan", [])

    counter = 1

    for step in plan:
        tool = step["tool"]
        query = step["query"]
        reason = step["reason"]

        line = f"{counter}. **{tool}**(`{query}`) — {reason}"

        st.markdown(line)

        counter = counter + 1

    # Recommendation (after validation)
    st.subheader("Recommendation")
    st.markdown(final.get("recommendation", "—"))

    # Validation verdict
    v = final.get("validation", {})
    badge = "[OK]" if v.get("verdict") == "APPROVED" else "[WARN]"
    st.subheader("Validation")
    st.markdown(f"{badge} **{v.get('verdict', '—')}** · grounded={v.get('grounded')} · "
                f"revised={v.get('revised')}")
    for issue in v.get("issues", []):
        st.markdown(f"- {issue}")

    # Full reasoning trace + evidence pool (great for the oral defense)
    trace = final.get("trace", [])

    with st.expander("Agent reasoning trace"):
        for line in trace:
            st.text(line)


    evidence = final.get("evidence", [])

    with st.expander("Evidence pool used by the agent"):
        for e in evidence:
            evidence_id = e["id"]
            source = e["source"]
            score = e["score"]
            title = e["title"]
            url = e["url"]

            line = (
                f"- **Evidence {evidence_id}** · {source} · Score {score}  \n"
                f"  {title}  \n"
                f"  {url}"
            )

            st.markdown(line)
