# Simulate offline functionality
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3
import json

# Constant for title-only filters
MIN_EVIDENCE_LENGTH = 80

# Vectors from db
db = sqlite3.connect("vectors.db")
lines = db.execute(
    "SELECT title, text, source, date, url, vector FROM docs").fetchall()
title = []
for z in lines:
    title.append(z[0])      # z[0] = Title

texts = []
for z in lines:
    texts.append(z[1])      # z[1] = Text

sources = []
for z in lines:
    sources.append(z[2])    # z[2] = Source

urls = []
for z in lines:
    urls.append(z[4])       # z[4] = URL
vectors = np.array([json.loads(z[5]) for z in lines])

# semantic search
model = SentenceTransformer("intfloat/multilingual-e5-base")


def search(question, k=5):
    # The model expects the prefix "query: " in search queries
    # Question -> Vector
    f = model.encode("query: " + question, normalize_embeddings=True)

    # Cosine (vectors are normalised)
    similarity = []
    for v in vectors:
        similarity.append(np.dot(v, f))
    similarity = np.array(similarity)

    # sort by similarity
    sorted = similarity.argsort()   # klein -> gross
    desc = sorted[::-1]             # gross -> klein

    # Title-Only-Filter
    results = []
    for i in desc:
        
        if len(texts[i].strip()) < MIN_EVIDENCE_LENGTH:
            continue
        results.append((title[i], texts[i], sources[i], urls[i]))
        if len(results) >= k:
            break

    return results


# Questions
# question = "Was ist aktuell das größte Risiko für Lufthansa, und welche Belege stützen das?"
# question = "What is currently the biggest risk for Lufthansa, and what evidence supports it?"

# question = "Welche strategische Chance sollte Lufthansa als Nächstes priorisieren?"
# question = "Which strategic opportunity should Lufthansa prioritize next?"

# question = "Welche Maßnahmen sollten aus heutiger Sicht priorisiert werden und warum?"
# question = "From today’s perspective, which measures should be prioritised, and why?"

# question = "Welche Aktivitäten der Wettbewerber sollte Lufthansa aktuell im Blick behalten und welche Belege zeigen das?"
# question = "Which competitor activities should Lufthansa monitor right now, and what evidence shows this?"

# question = "Welche Technologien oder Branchentrends sollte das Lufthansa-Management beobachten – gestützt auf die aktuellen Meldungen?"
question = "Which technologies or industry trends should Lufthansa's management monitor, based on the current reports?"

list_of_results = search(question, k=5)

evidence = ""
number = 1
for match in list_of_results:
    # match ist ein Tupel: (Titel, Text, Quelle, URL)
    text = match[1]       # zweites Feld = Text
    source = match[2]     # drittes Feld = Quelle
    url = match[3]        # viertes Feld = URL

    text_short = text[:300]   # nur die ersten 300 Zeichen
    line = "Evidence " + str(number) + ": " + text_short \
        + " (Source: " + source + " - " + url + ")"
    evidence = evidence + line + "\n"
    number = number + 1

# Prompt bauen (Rolle + Belege + Struktur)
prompt = f"""You are a strategy adviser to the Lufthansa Executive Board.
Answer the question using ONLY the evidence below. Do not invent facts, figures or events
that are not in the evidence. You MAY draw reasonable strategic conclusions from what the
evidence reports - that is your job. Reply with exactly "INSUFFICIENT EVIDENCE" ONLY if the
evidence is unrelated to the question or contains no usable information.

Question: {question}

Evidence:
{evidence}
Please reply using exactly this structure:
RECOMMENDATION: <a specific action, or "INSUFFICIENT EVIDENCE">
SUPPORTING DOCUMENTS: <the document numbers you are referring to>
EXPECTED IMPACT: <benefits for the company>
RISK: <financial/operational/strategic risk>
"""

# LLM-Call
answer = ollama.chat(model="qwen3:8b",
                     messages=[{"role": "user", "content": prompt}])
print(answer["message"]["content"])

# Evidence
print("\nEvidence")
number = 1
for match in list_of_results:
    title_doc = match[0]
    source = match[2]
    url = match[3]
    print("Evidence " + str(number) + ": " +
          title_doc + " (" + source + " - " + url + ")")
    number = number + 1
