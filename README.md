# StockCast

A stock analysis dashboard that fetches historical price data and generates multi-day forecasts using machine learning.

## Features

- **1 year of daily OHLCV data** from Yahoo Finance via yfinance (no API key needed)
- **ML-powered forecasting** using Random Forest with engineered technical features (SMA, RSI, momentum, volatility)
- **5-day price forecast** with confidence intervals derived from tree variance
- **Technical indicator overlays** — SMA(5), SMA(20), and RSI(14) rendered on the chart
- **Interactive candlestick chart** built with TradingView Lightweight Charts, with synced RSI sub-chart
- **Watchlist** — search multiple tickers and switch between them; persists across page reloads via localStorage
- **Server-side caching** with 15-minute TTL to avoid redundant fetches
- **Model accuracy metrics** — MAE (%) and directional accuracy computed via walk-forward evaluation

## Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn (RandomForestRegressor), pandas, NumPy
- **Data:** yfinance
- **Frontend:** Vanilla HTML/CSS/JS, TradingView Lightweight Charts, DM Sans/DM Mono (Google Fonts)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Open `http://localhost:5003/`

## Usage

Enter a ticker symbol (e.g. AAPL, MSFT, GOOGL) or click one of the suggestions. The dashboard shows:

- Current price and daily change
- ML-predicted next-day price with confidence range
- Model accuracy (MAE % and directional accuracy)
- Candlestick chart with SMA(5), SMA(20) overlays and volume
- 5-day forecast line with confidence bounds
- RSI(14) sub-chart with overbought/oversold reference lines

Search multiple tickers to build a watchlist. Click any card to switch views. Hover a card for refresh and remove buttons.

## Model Evaluation

Run the standalone evaluation script to test model accuracy across multiple tickers:

```bash
python evaluate_model.py
```
