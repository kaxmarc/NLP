import feedparser
import json
import re

# Feeds relating to "Lufthansa"
LUFTHANSA_ONLY_FEEDS = [
    ("Google News DE",
     "https://news.google.com/rss/search?q=Lufthansa&hl=de&gl=DE&ceid=DE:de"),
    ("Bing News",             "https://www.bing.com/news/search?q=Lufthansa&format=rss"),
    ("Reddit r/lufthansa",    "https://www.reddit.com/r/lufthansa/.rss"),
    ("Yahoo Finance LHA.DE",
     "https://feeds.finance.yahoo.com/rss/2.0/headline?s=LHA.DE&region=DE&lang=de-DE"),
    ("Yahoo Finance DLAKY",
     "https://feeds.finance.yahoo.com/rss/2.0/headline?s=DLAKY&region=US&lang=en-US"),

    # An alternative solution to the BeautifulSoup 403 problem.
    ("Google News (official)",
     "https://news.google.com/rss/search?q=Lufthansa+site:lufthansagroup.com&hl=en-US&gl=US&ceid=US:en"),

]

# Broad industry feeds (only entries relating to Lufthansa)
GLOBAL_PRESS = [
    ("aeroTELEGRAPH",         "https://www.aerotelegraph.com/feed"),
    ("Aviation Week",         "https://aviationweek.com/rss.xml"),
]

# Feeds on key competitors
COMPETITOR_FEEDS = [
    ("Google News DE - Ryanair",
     "https://news.google.com/rss/search?q=Ryanair&hl=de&gl=DE&ceid=DE:de"),
    ("Google News DE - Wizz Air",
     "https://news.google.com/rss/search?q=%22Wizz+Air%22&hl=de&gl=DE&ceid=DE:de"),
    ("Google News DE - easyJet",
     "https://news.google.com/rss/search?q=easyJet&hl=de&gl=DE&ceid=DE:de"),
    ("Google News DE - Air France-KLM",
     "https://news.google.com/rss/search?q=%22Air+France%22+KLM&hl=de&gl=DE&ceid=DE:de"),
    ("Google News DE - British Airways/IAG",
     "https://news.google.com/rss/search?q=%22British+Airways%22+OR+IAG&hl=de&gl=DE&ceid=DE:de"),
    ("Google News DE - Turkish Airlines",
     "https://news.google.com/rss/search?q=%22Turkish+Airlines%22&hl=de&gl=DE&ceid=DE:de"),
    ("Google News EN - Emirates",
     "https://news.google.com/rss/search?q=%22Emirates+airline%22&hl=en-US&gl=US&ceid=US:en"),
    ("Google News EN - Qatar Airways",
     "https://news.google.com/rss/search?q=%22Qatar+Airways%22&hl=en-US&gl=US&ceid=US:en"),
]

# Lufthansa Group including subsidiaries
KEYWORDS = re.compile(
    r"lufthansa|\bswiss\b|austrian airlines|brussels airlines|eurowings|"
    r"discover airlines|lufthansa cargo|lufthansa technik|ITA airways",
    re.I,
)

# Competitors including subsidiaries
COMPETITOR_KEYWORDS = re.compile(
    r"ryanair|wizz ?air|easyjet|british airways|\bIAG\b|iberia|air france|"
    r"\bKLM\b|turkish airlines|emirates|qatar airways",
    re.I,
)
TAGS = re.compile(r"<[^>]+>")


def strip_html(html):
    return TAGS.sub("", html or "").strip()


def select_best_text(e):
    longest = e.get("summary", "")
    for c in e.get("content", []):
        if len(c.get("value", "")) > len(longest):
            longest = c["value"]
    return strip_html(longest)


docs = []
seen = set()


def collect(source_name, url, pattern=None):
    # Read the feed/news
    feed = feedparser.parse(url)
    added_count = 0
    for e in feed.entries:
        if "title" in e:
            title = e["title"]
        else:
            title = ""
        # content > summary
        text = select_best_text(e)
        # pattern set -> keep only matching entries (otherwise all)
        if pattern and not pattern.search(title + " " + text):
            continue
        if "link" in e:
            link = e["link"]
        else:
            link = ""
        # Check for duplicates
        if not link or link in seen:
            continue
        seen.add(link)
        # Add feed/news
        docs.append({
            "source": source_name,
            "title": title,
            "text": text,
            "date": e.get("published", ""),
            "url": link,
        })
        added_count += 1
    print(f"  {added_count:4d}  {source_name}")


print("\nLufthansa Feeds:")
for name, url in LUFTHANSA_ONLY_FEEDS:
    collect(name, url, pattern=None)

print("\nLufthansa press releases:")
for name, url in GLOBAL_PRESS:
    collect(name, url, pattern=KEYWORDS)

print("\nCompetitors Feeds:")
for name, url in COMPETITOR_FEEDS:
    collect(name, url, pattern=None)

# Domain feeds competitors
print("\nCompetitors in the trade press:")
for name, url in GLOBAL_PRESS:
    collect(name + " (nompetitors)", url, pattern=COMPETITOR_KEYWORDS)

json.dump(docs, open("raw_data.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

sources_count = len({d["source"] for d in docs})
print(f"\n{len(docs)} Documents collected from {sources_count} independent sources.")
