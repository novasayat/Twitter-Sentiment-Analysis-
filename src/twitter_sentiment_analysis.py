"""Twitter sentiment analysis utilities.

This module provides a clean, testable pipeline for:
1. Loading tweets from either Twitter API or CSV input.
2. Cleaning tweet text.
3. Computing polarity and subjectivity with TextBlob.
4. Classifying tweets into Positive/Neutral/Negative sentiment buckets.
5. Exporting summary artifacts for reporting.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
from wordcloud import STOPWORDS, WordCloud

try:
    import tweepy
except ImportError:  # tweepy is optional when working with local CSV files
    tweepy = None


CLEANING_PATTERNS = [
    (r"@[A-Za-z0-9_]+", ""),  # @mentions
    (r"#", ""),  # hashtag symbol only (keep the word)
    (r"RT\\s+", ""),  # retweet marker
    (r"https?://\\S+", ""),  # URLs
    (r"\\n", " "),  # new lines
    (r"\\s+", " "),  # repeated spaces
]


@dataclass(frozen=True)
class AnalysisConfig:
    """Runtime configuration for sentiment analysis."""

    max_tweets: int = 200
    output_dir: Path = Path("outputs")
    chart_style: str = "fivethirtyeight"


def clean_text(text: str) -> str:
    """Normalize tweet text by removing common Twitter noise."""

    cleaned = text
    for pattern, replacement in CLEANING_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned.strip()


def polarity(text: str) -> float:
    """Return polarity score in the range [-1, 1]."""

    return TextBlob(text).sentiment.polarity


def subjectivity(text: str) -> float:
    """Return subjectivity score in the range [0, 1]."""

    return TextBlob(text).sentiment.subjectivity


def classify_sentiment(score: float) -> str:
    """Map polarity score to label."""

    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def fetch_recent_tweets(username: str, max_tweets: int, bearer_token: str) -> pd.DataFrame:
    """Fetch recent tweets using Twitter API v2 via Tweepy Client."""

    if tweepy is None:
        raise ImportError("tweepy is not installed. Install requirements to fetch from Twitter API.")

    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    user = client.get_user(username=username)
    if user.data is None:
        raise ValueError(f"User '{username}' was not found.")

    tweets = client.get_users_tweets(
        id=user.data.id,
        max_results=min(max_tweets, 100),
        exclude=["retweets", "replies"],
        tweet_fields=["lang", "created_at", "text"],
    )

    rows = []
    for tweet in tweets.data or []:
        rows.append(
            {
                "created_at": tweet.created_at,
                "text": tweet.text,
            }
        )

    return pd.DataFrame(rows)


def load_tweets_from_csv(csv_path: Path, text_column: str = "text") -> pd.DataFrame:
    """Load tweets from CSV and normalize column names."""

    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {csv_path}")

    return df.rename(columns={text_column: "text"})[["text"]].copy()


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning and sentiment scoring to a tweets dataframe."""

    result = df.copy()
    result["clean_text"] = result["text"].astype(str).map(clean_text)
    result["subjectivity"] = result["clean_text"].map(subjectivity)
    result["polarity"] = result["clean_text"].map(polarity)
    result["sentiment"] = result["polarity"].map(classify_sentiment)
    return result


def sentiment_percentages(df: pd.DataFrame) -> pd.Series:
    """Return percentage distribution for sentiment labels."""

    distribution = df["sentiment"].value_counts(normalize=True).mul(100).round(2)
    return distribution.reindex(["Positive", "Neutral", "Negative"]).fillna(0)


def build_wordcloud(texts: Iterable[str], output_path: Path) -> None:
    """Generate and save word cloud image."""

    stopwords = set(STOPWORDS)
    stopwords.update({"co", "amp", "https", "t"})
    all_words = " ".join(texts)

    cloud = WordCloud(
        stopwords=stopwords,
        background_color="white",
        width=1200,
        height=600,
        random_state=42,
    ).generate(all_words)

    plt.figure(figsize=(12, 6))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_sentiment_bar_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Generate and save sentiment count bar chart."""

    plt.figure(figsize=(8, 5))
    df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).plot(kind="bar")
    plt.title("Tweet Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Tweet Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_pipeline(
    *,
    config: AnalysisConfig,
    csv_input: Path | None,
    username: str | None,
    bearer_token: str | None,
    text_column: str,
) -> Path:
    """Run full analysis and export outputs."""

    if csv_input is None and username is None:
        raise ValueError("Provide either --csv-input or --username.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use(config.chart_style)

    if csv_input is not None:
        raw_df = load_tweets_from_csv(csv_input, text_column=text_column)
    else:
        if not bearer_token:
            raise ValueError("--bearer-token is required when using --username.")
        raw_df = fetch_recent_tweets(username=username, max_tweets=config.max_tweets, bearer_token=bearer_token)

    analyzed = analyze_dataframe(raw_df)
    analyzed_path = config.output_dir / "analyzed_tweets.csv"
    analyzed.to_csv(analyzed_path, index=False)

    percentages = sentiment_percentages(analyzed)
    summary_path = config.output_dir / "sentiment_summary.csv"
    percentages.rename("percentage").to_csv(summary_path)

    build_wordcloud(analyzed["clean_text"], config.output_dir / "wordcloud.png")
    build_sentiment_bar_chart(analyzed, config.output_dir / "sentiment_bar_chart.png")

    return analyzed_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""

    parser = argparse.ArgumentParser(description="Professional Twitter sentiment analysis pipeline")
    parser.add_argument("--csv-input", type=Path, default=None, help="Path to CSV file with tweet text")
    parser.add_argument("--text-column", default="text", help="CSV column that contains tweet text")
    parser.add_argument("--username", default=None, help="Twitter username to fetch tweets from")
    parser.add_argument("--bearer-token", default=None, help="Twitter API bearer token")
    parser.add_argument("--max-tweets", type=int, default=200, help="Maximum tweets to fetch from API")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated files")
    return parser


def main() -> None:
    """CLI entrypoint."""

    args = build_arg_parser().parse_args()
    config = AnalysisConfig(max_tweets=args.max_tweets, output_dir=args.output_dir)

    output = run_pipeline(
        config=config,
        csv_input=args.csv_input,
        username=args.username,
        bearer_token=args.bearer_token,
        text_column=args.text_column,
    )
    print(f"Analysis complete. Results written to: {output}")


if __name__ == "__main__":
    main()
