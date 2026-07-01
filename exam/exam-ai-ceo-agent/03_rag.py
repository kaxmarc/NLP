# Simulate offline functionality
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import sqlite3
import collections
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Vectors from db
db = sqlite3.connect("vectors.db")
lines = db.execute(
    "SELECT title, text, source, date, url, vector FROM docs").fetchall()
title = []
for z in lines:
    title.append(z[0])      # z[0] = field title

texts = []
for z in lines:
    texts.append(z[1])      # z[1] = field text

sources = []
for z in lines:
    sources.append(z[2])    # z[2] = field source
vectors = np.array([json.loads(z[5]) for z in lines])

# Semantic search
model = SentenceTransformer("intfloat/multilingual-e5-base")


def search(question, k=5):
    # e5 Prefix "query: "
    # Question to Vektor
    f = model.encode("query: " + question, normalize_embeddings=True)
    similarity = []
    for v in vectors:
        similarity.append(np.dot(v, f))
    similarity = np.array(similarity)
    sorted = similarity.argsort()   # small -> large
    desc = sorted[::-1]             # large -> small
    top = desc[:k]                  # the best k
    return [(title[i], texts[i], sources[i], float(similarity[i])) for i in top]


print("Top results for the risk question:")
list_of_results = search("Which customer-related risk is the greatest?")

for match in list_of_results:
    title = match[0]
    source = match[2]
    score = match[3]

    score_rounded = round(score, 2)
    title_short = title[:70]
    source_scope = "[" + source + "]"

    print(score_rounded, "-", source_scope, title_short)

# Sentiment
mood = pipeline("sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                truncation=True, max_length=512)


def label(text):
    result = mood(text)
    # {'label': '5 stars', 'score': 0.485652357339859}
    first_result = result[0]
    label_text = first_result["label"]      # 5 stars
    first_sign = label_text[0]
    stars = int(first_sign)                 # 5
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


# Mood
counter = collections.Counter(label(t) for t in texts if t.strip())
print("Snapshot of the mood:", dict(counter))
