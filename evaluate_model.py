import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data.data_fetcher import fetch_stock_data


def predict_price(historical_data, days_back=30):
    closing_prices = historical_data['close'].tail(days_back).values
    
    X = np.arange(len(closing_prices)).reshape(-1, 1)
    y = closing_prices
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_day = np.array([[len(closing_prices)]])
    prediction = model.predict(next_day)[0]
    
    return prediction


def evaluate_model(symbol, test_days=10):
    print(f"Evaluating model for {symbol}...")
    print("-" * 50)
    
    df = fetch_stock_data(symbol)
    
    if len(df) < 40:
        print(f"Error: Not enough data (need at least 40 days, got {len(df)})")
        return
    
    predictions = []
    actual_prices = []
    
    for i in range(test_days):
        test_date_index = len(df) - test_days + i
        if test_date_index < 30:
            continue
        
        historical_data = df.iloc[:test_date_index]
        actual_price = df.iloc[test_date_index]['close']
        
        prediction = predict_price(historical_data)
        
        predictions.append(prediction)
        actual_prices.append(actual_price)
    
    predictions = np.array(predictions)
    actual_prices = np.array(actual_prices)
    
    mae = mean_absolute_error(actual_prices, predictions)
    mse = mean_squared_error(actual_prices, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_prices, predictions)
    
    avg_price = np.mean(actual_prices)
    mae_percent = (mae / avg_price) * 100
    
    print(f"\nResults for {symbol}:")
    print(f"Tested on {len(predictions)} days")
    print(f"\nAccuracy Metrics:")
    print(f"  Mean Absolute Error (MAE): ${mae:.2f} ({mae_percent:.2f}% of avg price)")
    print(f"  Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"  R-squared (R²): {r2:.4f}")
    print(f"\n  Average Actual Price: ${avg_price:.2f}")
    print(f"  Average Prediction: ${np.mean(predictions):.2f}")
    
    print(f"\nSample Predictions (last 5 days):")
    print(f"{'Date':<12} {'Actual':<10} {'Predicted':<12} {'Error':<10} {'Error %':<10}")
    print("-" * 60)
    
    for i in range(max(0, len(predictions)-5), len(predictions)):
        error = predictions[i] - actual_prices[i]
        error_pct = (error / actual_prices[i]) * 100
        date = df.iloc[len(df) - len(predictions) + i]['date'].strftime('%Y-%m-%d')
        print(f"{date:<12} ${actual_prices[i]:<9.2f} ${predictions[i]:<11.2f} ${error:<9.2f} {error_pct:<9.2f}%")
    
    print("\n" + "=" * 50)
    print("Interpretation:")
    print(f"  MAE of ${mae:.2f} means predictions are off by ${mae:.2f} on average")
    print(f"  R² of {r2:.4f} means model explains {r2*100:.2f}% of price variance")
    if r2 > 0.5:
        print("  ✓ Model has decent predictive power")
    elif r2 > 0:
        print("  ⚠ Model has weak predictive power")
    else:
        print("  ✗ Model performs worse than just using the average price")


if __name__ == "__main__":
    print("=" * 50)
    print("Stock Prediction Model Evaluation")
    print("=" * 50)
    print("\nThis script tests the model's accuracy by:")
    print("1. Using historical data up to a certain date")
    print("2. Making a prediction for the next day")
    print("3. Comparing the prediction to the actual price")
    print("4. Calculating accuracy metrics\n")
    
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        try:
            evaluate_model(symbol, test_days=10)
            print()
        except Exception as e:
            print(f"Error evaluating {symbol}: {str(e)}\n")

