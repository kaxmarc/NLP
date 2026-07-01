# Simulate offline functionality (models are already cached locally)
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
 
import json
import sqlite3
import collections
 
from sentence_transformers import SentenceTransformer
from transformers import pipeline
 
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
 
from agent_core import build_agent
 
LLM_MODEL = "qwen3:8b"
MIN_EVIDENCE_LENGTH = 80     # skip title-only entries (same rule as the RAG pipeline)

# e5 embeddings for LangChain
# The documents have been embedded since Step 2 and are simply reused here – this class 
# is needed to embed the search query at runtime (and to fully comply with LangChain’s embeddings interface).
class E5Embeddings(Embeddings):
 
    def __init__(self):
        self.model = SentenceTransformer("intfloat/multilingual-e5-base")
 
    def embed_documents(self, texts):
        return self.model.encode(["passage: " + t for t in texts],
                                 normalize_embeddings=True).tolist()
 
    def embed_query(self, text):
        return self.model.encode("query: " + text,
                                 normalize_embeddings=True).tolist()

# build FAISS 
# from stored vectors (Pre-computed e5 vectors are taken from vectors.db)
db = sqlite3.connect("vectors.db")
rows = db.execute(
    "SELECT title, text, source, date, url, vector FROM docs").fetchall()
 
text_embeddings, metadatas = [], []
for title, text, source, date, url, vector in rows:
    # Entries that are too short (title only, < 80 characters) are skipped
    if len(text.strip()) < MIN_EVIDENCE_LENGTH:
        continue
    text_embeddings.append((text, json.loads(vector)))
    metadatas.append({"title": title, "source": source, "url": url, "date": date})
 
embeddings = E5Embeddings()
vectorstore = FAISS.from_embeddings(text_embeddings, embeddings, metadatas=metadatas)
print(f"FAISS index built: {len(metadatas)} documents")
 
# Sentiment model (nlptown)
mood = pipeline("sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                truncation=True, max_length=512)


# LangChain @tool tools
# tool: search_evidence
# retrieves 5 documents
@tool
def search_evidence(query: str, k: int = 5):
    """Semantic similarity search over the Lufthansa news/RAG knowledge base; returns the top-k documents with title, text, source, url and cosine score."""
    # search_evidence: FAISS similarity search. The cosine is reconstructed from the L2 distance (cos = 1 - dist/2), because the vectors have length 1
    out = []

    results = vectorstore.similarity_search_with_score(query, k=k)

    for pair in results:
        doc = pair[0]
        dist = pair[1]

        cos = 1.0 - dist / 2.0
        cos = round(cos, 3)

        m = doc.metadata

        title = m.get("title", "")
        text = doc.page_content
        source = m.get("source", "")
        url = m.get("url", "")

        result = {
            "title": title,
            "text": text,
            "source": source,
            "url": url,
            "score": cos
        }

        out.append(result)

    return out
 
# tool: analyze_sentiment
# retrieves 8 documents and measures the sentiment
@tool
def analyze_sentiment(query: str, k: int = 8):
    """Retrieve documents for a topic and measure public sentiment (positive/neutral/negative)."""
    parameter = {"query": query, "k": k}
    hits = search_evidence.invoke(parameter)
    dist = sentiment_of_texts([h["text"] for h in hits])
    # return both: the sentiment AND the documents (so the agent can keep them as evidence)
    return {"sentiment": dist, "hits": hits}
 
# sentiment_of_texts: converts the nlptown star ratings (1–5) to negative/neutral/positive
def sentiment_of_texts(text_list):
    def label(text):
        truncated_text = text[:512]
        result = mood(truncated_text)
        first_result = result[0]
        label = first_result["label"]
        first_char = label[0]
        stars = int(first_char)
        if stars <= 2:
            return "negative"
        if stars == 3:
            return "neutral"
        return "positive"
    return dict(collections.Counter(
        label(t) for t in text_list if t and t.strip()))

# route the graph through the tools (Tuple processing), search tool
def search(query, k=5):
    docs = search_evidence.invoke({"query": query, "k": k})
    return [(d["title"], d["text"], d["source"], d["url"], d["score"]) for d in docs]

# route the graph through the analyze_sentiment tool: returns the sentiment AND the docs as tuples
def analyze(query, k=8):
    result = analyze_sentiment.invoke({"query": query, "k": k})
    dist = result["sentiment"]
    docs = result["hits"]
    hits = [(d["title"], d["text"], d["source"], d["url"], d["score"]) for d in docs]
    return dist, hits


if __name__ == "__main__":
    # ChatOllama (qwen3:8b) is generated and passed to build_agent together with search + sentiment + analyze
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    agent = build_agent(llm, search, sentiment_of_texts, analyze)
 
    # USER GOAL DEFINITION
    goal = ("If you were the CEO today, what would you do next and why?")
    # goal = ("What is currently the biggest strategic risk for Lufthansa, "
    #        "and what should the board do about it?")
    print("GOAL:", goal)
    print("Running LangGraph agent (Plan -> Retrieve -> Analyze -> Decide -> Validate) ...\n")
 
    # agent.invoke({'goal': ...}) starts the entire workflow and returns the final state
    state = agent.invoke({"goal": goal})
 
    print("\n--- PLAN --------------------------------")
    for i, step in enumerate(state["plan"], 1):
        print(f"  {i}. {step['tool']}('{step['query']}')  -- {step['reason']}")
 
    print("\n--- TRACE -------------------------------")
    for line in state["trace"]:
        print("  " + line)
 
    print("\n--- EVIDENCE POOL -----------------------")
    for e in state["evidence"]:
        print(f"  Evidence {e['id']} [score {e['score']}] {e['source']}: {e['title'][:70]}")
 
    print("\n--- RECOMMENDATION ----------------------")
    print(state["recommendation"])
 
    print("\n--- VALIDATION --------------------------")
    v = state["validation"]
    print(f"  Verdict : {v['verdict']}   (grounded={v['grounded']}, revised={v['revised']})")
    for issue in v["issues"]:
        print("  Issue   : " + issue)

