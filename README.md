# FX Morning Report (Modern 2025 Institutional)

This project generates a **Modern 2025 Institutional‑style** daily FX update and weekly outlook. It features a high-density, authoritative design inspired by top-tier investment banking reports (CIBC, Goldman Sachs).

> ⚠️ Disclaimer: This is for information/education only and **not** investment advice.  
> Data quality varies by source (yfinance is an unofficial wrapper around Yahoo Finance data and may be delayed).

## Key Features

- **Markets Dashboard**: A high-density cross-asset command center (`index.html`).
- **Institutional Design**: Maroon-themed UI (#8B0021) with professional **Outfit/Lora** typography.
- **Dynamic Generation**: The local server automatically regenerates the report every time you refresh the page.
- **Interactive Charts**: Responsive Plotly-based line charts and Relative Rotation Graphs (RRG).
- **Public Hosting Ready**: Integrated GitHub Actions for automatic daily updates on GitHub Pages.

## What it generates

- `site/index.html` (Markets Dashboard | Institutional Intelligence)
- `site/daily.html` (Daily FX Update)
- `site/weekly.html` (Weekly FX Outlook | RRG Chart)
- `site/crypto.html` (Crypto Pulse)
- Interactive Plotly charts embedded directly in the HTML.

## Data sources

- **FX spot & OHLC**: `yfinance` or **Interactive Brokers (IBKR)**.
- **Cross‑asset**: DXY proxy, S&P 500, WTI, Gold, VIX, etc. via `yfinance`.
- **U.S. rates**: U.S. Treasury yield curve and analysis.
- **Positioning**: CFTC Commitment of Traders (COT) with sentiment mapping.
- **Street forecasts**: Aggregated from Scotiabank, TD Economics, MUFG, and NBC.

## Quick start (macOS)

```bash
# 1) install deps
pip install -r requirements.txt

# 2) generate the report manually
python -m fxmorning.cli generate

# 3) serve with dynamic refresh (http://127.0.0.1:8000)
python -m fxmorning.cli serve --port 8000
```

## Production Deployment

This project includes a built-in **GitHub Actions** workflow at `.github/workflows/deploy.yml`. 

1. Push this project to a GitHub repository.
2. The Action will automatically run every weekday morning (8 AM ET).
3. It will generate the reports and host them for free on **GitHub Pages**.

## Configuration

Edit `config.yaml` to customize:
- `pairs`: Currency codes and tickers.
- `data_source`: Switch between `yfinance` and `ibkr`.
- `editorial`: Add your own macro views and expert quotes.
- `timezone`: Set your local reporting timezone.
