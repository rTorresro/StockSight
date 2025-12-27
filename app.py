from flask import Flask, jsonify, render_template
from data.data_fetcher import fetch_stock_data
from model import predict_price
from dotenv import load_dotenv
import os 

app = Flask(__name__)

load_dotenv() 

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "1HQHBPL1U7I1NGI9")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stock/<symbol>")
def get_stock(symbol):
    try:
        symbol = symbol.upper()
        
        df = fetch_stock_data(symbol, API_KEY)
        
        current_price = float(df['close'].iloc[-1])
        
        previous_price = float(df['close'].iloc[-2])
        
        change = current_price - previous_price
        change_percent = (change / previous_price) * 100
        
        prediction = float(predict_price(df))
        
        return jsonify({
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'change': round(change, 2),
            'change_percent': round(change_percent, 2),
            'prediction': round(prediction, 2)
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to fetch data: {str(e)}'}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5002)
