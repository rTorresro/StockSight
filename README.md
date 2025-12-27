# Stock Prediction Platform

A full-stack web application that fetches real-time stock data and predicts future prices using machine learning.

## Features

- Real-time stock price data from Alpha Vantage API
- ML-powered price prediction using linear regression
- Interactive web dashboard
- RESTful API endpoints

## Technologies

- Python
- Flask
- scikit-learn
- Pandas
- NumPy
- JavaScript
- HTML/CSS

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get Alpha Vantage API key from [alphavantage.co](https://www.alphavantage.co/support/#api-key)

3. Create a `.env` file (optional):
   ```
   ALPHA_VANTAGE_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open browser to `http://localhost:5002/`

## Usage

Enter a stock symbol (e.g., AAPL, MSFT, GOOGL) and click "Get Stock Data" to see:
- Current stock price
- 24-hour price change
- ML-predicted next day price

