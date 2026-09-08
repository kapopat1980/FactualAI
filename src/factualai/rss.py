"""Real-time RSS live feed classification (Section 5.5 of the paper).

Retrieves the most recent headlines/summaries from a fixed set of news
outlets and classifies them through the Random Forest pipeline only (the LLM
is omitted here to preserve responsiveness, per the paper).
"""
from __future__ import annotations

from dataclasses import dataclass

import feedparser

from .classifier import RandomForestNewsClassifier

DEFAULT_FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "Reuters": "https://feeds.reuters.com/reuters/topNews",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
    "NDTV": "https://feeds.feedburner.com/ndtvnews-top-stories",
}

HEADLINES_PER_SOURCE = 4


@dataclass
class RssItem:
    source: str
    title: str
    summary: str
    label: str
    confidence: float


def fetch_headlines(feeds: dict[str, str] = DEFAULT_FEEDS) -> list[dict]:
    """Fetch the most recent headlines/summaries per source."""
    items = []
    for source, url in feeds.items():
        parsed = feedparser.parse(url)
        for entry in parsed.entries[:HEADLINES_PER_SOURCE]:
            items.append(
                {
                    "source": source,
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", entry.get("title", "")),
                }
            )
    return items


def classify_live_feed(
    classifier: RandomForestNewsClassifier, feeds: dict[str, str] = DEFAULT_FEEDS
) -> list[RssItem]:
    """Fetch headlines and classify each with the Random Forest pipeline only."""
    results = []
    for item in fetch_headlines(feeds):
        text = f"{item['title']}. {item['summary']}"
        fake_score = classifier.rf_fake_score(text)
        label = "Fake" if fake_score >= 0.5 else "Real"
        confidence = fake_score if label == "Fake" else 1.0 - fake_score
        results.append(
            RssItem(
                source=item["source"],
                title=item["title"],
                summary=item["summary"],
                label=label,
                confidence=confidence,
            )
        )
    return results
