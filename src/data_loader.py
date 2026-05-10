from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd
import yfinance as yf


def download_stock_data(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data found for symbol: {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename_axis("Date").reset_index()
    data["Ticker"] = ticker.upper()
    return data


def download_intraday_data(ticker: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
    """Download intraday OHLCV data for short-range charting."""
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No intraday data found for symbol: {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename_axis("Date").reset_index()
    data["Ticker"] = ticker.upper()
    return data


def resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    monthly = (
        data.set_index("Date")
        .resample("ME")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
                "Ticker": "last",
            }
        )
        .dropna()
        .reset_index()
    )
    return monthly


def fetch_stock_news(ticker: str, limit: int = 12) -> pd.DataFrame:
    """Collect recent stock news from Yahoo Finance and Google News RSS."""
    articles = []
    seen = set()

    try:
        ticker_obj = yf.Ticker(ticker)
        yahoo_news = ticker_obj.news or []
    except Exception:
        yahoo_news = []

    for item in yahoo_news:
        content = item.get("content", item)
        title = content.get("title") or item.get("title")
        link = content.get("canonicalUrl", {}).get("url") or content.get("clickThroughUrl", {}).get("url") or item.get("link")
        provider = content.get("provider", {}).get("displayName") or item.get("publisher") or "Yahoo Finance"
        published = content.get("pubDate") or item.get("providerPublishTime")

        if isinstance(published, int):
            published_at = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()
        else:
            published_at = str(published or "")

        if title and link and link not in seen:
            articles.append(
                {
                    "title": title,
                    "source": provider,
                    "published": published_at,
                    "url": link,
                }
            )
            seen.add(link)

    query = quote_plus(f"{ticker} stock")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        request = Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=8) as response:
            root = ET.fromstring(response.read())

        for item in root.findall(".//item"):
            title = item.findtext("title")
            link = item.findtext("link")
            source = item.findtext("source") or "Google News"
            pub_date = item.findtext("pubDate") or ""
            try:
                published_at = parsedate_to_datetime(pub_date).isoformat()
            except Exception:
                published_at = pub_date

            if title and link and link not in seen:
                articles.append(
                    {
                        "title": title,
                        "source": source,
                        "published": published_at,
                        "url": link,
                    }
                )
                seen.add(link)
    except Exception:
        pass

    return pd.DataFrame(articles[:limit])
