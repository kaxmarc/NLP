# Simulate offline functionality
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import re
import sqlite3
from sentence_transformers import SentenceTransformer

# load raw data
docs = json.load(open("raw_data.json", encoding="utf-8"))

# cleaning part
def clean(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

cleaned = []
seen = set()
seen_titles = set()

for d in docs:
    titel = clean(d["title"])
    text = clean(d["text"])
    content = (titel + " " + text).strip()
    if len(content) < 40:
        continue
    if content.lower() in seen:
        continue
    titel_key = re.sub(r"\W+", " ", titel.lower()).strip()[:70]
    if titel_key and titel_key in seen_titles:
        continue
    seen_titles.add(titel_key)
    seen.add(content.lower())
    cleaned.append({"title": titel, "text": text, "content": content,
                   "source": d.get("source", ""),
                    "date": d["date"], "url": d["url"]})
print(len(cleaned), "clean documents")

# embedding
modell = SentenceTransformer("intfloat/multilingual-e5-base")
texte = ["passage: " + d["content"] for d in cleaned]
vectors = modell.encode(texte, normalize_embeddings=True)

# storing (vectors in db)
db = sqlite3.connect("vectors.db")
db.execute("DROP TABLE IF EXISTS docs")
db.execute("""CREATE TABLE docs
              (id INTEGER PRIMARY KEY, title TEXT, text TEXT, source TEXT,
               date TEXT, url TEXT, vector TEXT)""")
for d, v in zip(cleaned, vectors):
    db.execute("INSERT INTO docs (title, text, source, date, url, vector) VALUES (?,?,?,?,?,?)",
               (d["title"], d["text"], d["source"], d["date"], d["url"], json.dumps(v.tolist())))
db.commit()
db.close()
print("Saved in vectors.db")
