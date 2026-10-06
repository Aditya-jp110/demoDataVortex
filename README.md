# Multi-Agent Big Data Analytics Platform — Local Prototype

## What it demonstrates

1. Mock ingestion of structured daily business metrics + unstructured JSON-like logs.
2. A simulated PySpark-style processing layer with cleaning, rolling averages and anomaly tagging.
3. A modular multi-agent swarm:
   - Data Query Agent
   - Pattern & Trend Discovery Agent
   - Predictive ML Agent
   - Synthesizer / Reporting Agent
4. Natural-language query routing.
5. Interactive Streamlit dashboard with Plotly history + forecast chart.
6. Executive Markdown report.

## Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

No cloud credentials, database server, Spark cluster, or API key is required.

## Example query

`Find the revenue drop cause and predict the next 3 days`

Other examples:
- `Show average and highest revenue`
- `Find anomaly cause`
- `Forecast next 7 days`
