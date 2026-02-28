# Twitter Sentiment Analysis

This project is now set up in a **notebook-first** way while still keeping a modern code structure in the repository.

## What you get

- `Twitter_Sentiment_Analysis.ipynb` as the main, easy-to-follow workflow.
- A default **demo mode** that runs without Twitter credentials using local sample data.
- Optional **Twitter API mode** for live tweets with your own bearer token.
- Modern helper module and tests (`src/`, `tests/`) for maintainability.

## Repository structure

```text
.
├── Twitter_Sentiment_Analysis.ipynb   # Main notebook (recommended start)
├── data/
│   └── sample_tweets.csv              # Credential-free demo data
├── src/
│   └── twitter_sentiment_analysis.py  # Modern reusable pipeline
├── tests/
│   └── test_sentiment_pipeline.py
├── requirements.txt
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open `Twitter_Sentiment_Analysis.ipynb`.

## Notebook modes

Inside the notebook:

- `USE_TWITTER_API = False` (default)
  - Uses `data/sample_tweets.csv`
  - Works without any credentials
  - Good for viewers who just want to see outputs/insights

- `USE_TWITTER_API = True`
  - Fetches live tweets from Twitter API
  - Requires your own bearer token in environment:

```bash
export TWITTER_BEARER_TOKEN="your_token_here"
```

## Note on security

Do not hardcode API keys/tokens in notebook or source files. Use environment variables.

## If you only care about the old one-file flow

Open `Twitter_Sentiment_Analysis.ipynb` and run it directly.
You can ignore `src/` and `tests/` completely if you only want notebook usage.
