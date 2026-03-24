import yfinance as yf
import pandas as pd


def fetch_stock_data(symbol, period='1y'):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)

    if df.empty:
        raise ValueError(
            f"No data found for '{symbol}'. Check the ticker symbol and try again."
        )

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df['date'] = pd.to_datetime(df['date'].dt.date)
    df = df.sort_values('date').reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = fetch_stock_data('AAPL')
    print(df)
