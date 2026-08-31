"""
Test: Can we fetch full article text from each of our 4 RSS sources?
Uses requests to download + trafilatura to extract clean text.
"""
import requests
import feedparser
import trafilatura

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

feeds = [
    ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("MIT Tech Review", "https://www.technologyreview.com/feed/"),
    ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ("Ars Technica AI", "https://feeds.arstechnica.com/arstechnica/technology-lab"),
]

for source_name, feed_url in feeds:
    print(f"\n{'='*60}")
    print(f"SOURCE: {source_name}")
    print(f"{'='*60}")

    # Get first article URL from RSS
    resp = requests.get(feed_url, headers=HEADERS, verify=False, timeout=10)
    parsed = feedparser.parse(resp.text)

    if not parsed.entries:
        print("  No entries found in feed!")
        continue

    article_url = parsed.entries[0].get("link", "")
    article_title = parsed.entries[0].get("title", "No title")
    print(f"  Article: {article_title}")
    print(f"  URL: {article_url}")

    # Fetch HTML with requests (handles SSL proxy), then extract with trafilatura
    try:
        article_resp = requests.get(article_url, headers=HEADERS, verify=False, timeout=15)
        if article_resp.status_code != 200:
            print(f"  ✗ HTTP {article_resp.status_code}")
            continue

        text = trafilatura.extract(article_resp.text)
        if text:
            word_count = len(text.split())
            print(f"  ✓ Full text extracted: {word_count} words")
            print(f"  Preview (first 300 chars):")
            print(f"  {text[:300]}...")
        else:
            print("  ✗ trafilatura extraction returned None")
    except Exception as e:
        print(f"  ✗ Error: {e}")
