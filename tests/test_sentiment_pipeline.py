from pathlib import Path

import pandas as pd

from src.twitter_sentiment_analysis import (
    AnalysisConfig,
    analyze_dataframe,
    classify_sentiment,
    clean_text,
    run_pipeline,
    sentiment_percentages,
)


def test_clean_text_removes_twitter_noise():
    text = "RT @user: I love #Python https://example.com\n"
    cleaned = clean_text(text)
    assert "@user" not in cleaned
    assert "#" not in cleaned
    assert "https://" not in cleaned


def test_classify_sentiment_labels():
    assert classify_sentiment(0.1) == "Positive"
    assert classify_sentiment(-0.2) == "Negative"
    assert classify_sentiment(0.0) == "Neutral"


def test_analyze_dataframe_adds_expected_columns():
    df = pd.DataFrame({"text": ["I love this", "I hate this", "It is okay"]})
    analyzed = analyze_dataframe(df)
    assert {"clean_text", "subjectivity", "polarity", "sentiment"}.issubset(analyzed.columns)


def test_sentiment_percentages_shape():
    df = pd.DataFrame({"sentiment": ["Positive", "Positive", "Negative"]})
    result = sentiment_percentages(df)
    assert list(result.index) == ["Positive", "Neutral", "Negative"]
    assert result["Positive"] == 66.67


def test_pipeline_csv_input(tmp_path: Path):
    source = tmp_path / "tweets.csv"
    source.write_text("text\nI love this product\nI dislike this service\n", encoding="utf-8")

    out_dir = tmp_path / "outputs"
    output_csv = run_pipeline(
        config=AnalysisConfig(output_dir=out_dir),
        csv_input=source,
        username=None,
        bearer_token=None,
        text_column="text",
    )

    assert output_csv.exists()
    assert (out_dir / "sentiment_summary.csv").exists()
    assert (out_dir / "wordcloud.png").exists()
    assert (out_dir / "sentiment_bar_chart.png").exists()
