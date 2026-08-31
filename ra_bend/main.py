# FastAPI app — pipeline trigger endpoints + WebSocket for chatbot

#test
from agents.scout import fetch_rss_feeds

items = fetch_rss_feeds()
print(f"fetched {len(items)} items")
for item in items[:3] :
    print(item["title"])
    print(item["snippet"])