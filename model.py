import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def predict_price(historical_data):
    closing_prices = historical_data['close'].tail(30).values
    
    X = np.arange(len(closing_prices)).reshape(-1, 1)
    
    y = closing_prices
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_day = np.array([[len(closing_prices)]])
    prediction = model.predict(next_day)[0]
    
    return prediction


if __name__ == "__main__":
    from data.data_fetcher import fetch_stock_data
    
    api_key = '1HQHBPL1U7I1NGI9'
    df = fetch_stock_data('AAPL', api_key)
    
    prediction = predict_price(df)
    latest_price = df['close'].iloc[-1]
    
    print(f"Latest closing price: ${latest_price:.2f}")
    print(f"Predicted next day price: ${prediction:.2f}")
    print(f"Predicted change: ${prediction - latest_price:.2f}")
