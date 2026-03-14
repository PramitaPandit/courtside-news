import os
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import feedparser
import trafilatura

NEWS_DIR = "data/news"

RSS_FEEDS = [
    "http://feeds.bbci.co.uk/sport/tennis/rss.xml",
    "https://www.espn.com/espn/rss/tennis/news",
]

def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9-_ ]+", "", s).strip().replace(" ", "_")
    return (s[:80] if s else "article")

def extract_text_from_url(url: str) -> str | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    return text

def fetch_latest_news(max_items_per_feed: int = 10, hours: int = 72, sleep_s: float = 0.25) -> int:
    """
    Fetch recent tennis news from RSS feeds and save as .txt files into data/news.
    Returns number of saved articles.
    """
    os.makedirs(NEWS_DIR, exist_ok=True)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    saved = 0

    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)

        for entry in feed.entries[:max_items_per_feed]:
            link = entry.get("link")
            title = entry.get("title", "Tennis article")

            if not link:
                continue

            # Filter by date if available
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                pub_dt = datetime(*published[:6], tzinfo=timezone.utc)
                if pub_dt < cutoff:
                    continue

            # Extract article text
            text = extract_text_from_url(link)
            if not text or len(text) < 400:
                continue

            # Save to file
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            fname = f"{ts}_{safe_filename(title)}.txt"
            path = os.path.join(NEWS_DIR, fname)

            with open(path, "w", encoding="utf-8") as f:
                f.write(f"Title: {title}\nURL: {link}\n\n{text}")

            saved += 1
            time.sleep(sleep_s)

    return saved