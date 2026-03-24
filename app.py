from flask import Flask, jsonify, render_template, request
from data.data_fetcher import fetch_stock_data
from model import predict_price, forecast_prices, compute_indicators
from datetime import datetime, timedelta
import pandas as pd

app = Flask(__name__)

_cache = {}
_CACHE_TTL = timedelta(minutes=15)


def _get_cached(symbol):
    if symbol in _cache:
        data, ts = _cache[symbol]
        if datetime.now() - ts < _CACHE_TTL:
            return data
    return None


def _set_cache(symbol, data):
    _cache[symbol] = (data, datetime.now())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stock/<symbol>")
def get_stock(symbol):
    try:
        symbol = symbol.upper()
        refresh = request.args.get('refresh') == '1'

        if not refresh:
            cached = _get_cached(symbol)
            if cached:
                return jsonify(cached)

        df = fetch_stock_data(symbol)

        current_price = float(df['close'].iloc[-1])
        previous_price = float(df['close'].iloc[-2])
        change = current_price - previous_price
        change_percent = (change / previous_price) * 100

        result = predict_price(df)
        forecasts = forecast_prices(df, days=5)
        indicators = compute_indicators(df)

        last_date = df['date'].iloc[-1]
        forecast_list = []
        for f in forecasts:
            fdate = last_date + pd.tseries.offsets.BDay(f['day'])
            forecast_list.append({
                'date': fdate.strftime('%Y-%m-%d'),
                'price': round(f['prediction'], 2),
                'low': round(f['low'], 2),
                'high': round(f['high'], 2),
            })

        history = []
        for _, row in df.iterrows():
            history.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'open': round(row['open'], 2),
                'high': round(row['high'], 2),
                'low': round(row['low'], 2),
                'close': round(row['close'], 2),
                'volume': int(row['volume']),
            })

        response_data = {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'change': round(change, 2),
            'change_percent': round(change_percent, 2),
            'prediction': round(result['prediction'], 2),
            'prediction_low': round(result['low'], 2),
            'prediction_high': round(result['high'], 2),
            'mae_percent': result['mae_percent'],
            'direction_accuracy': result['direction_accuracy'],
            'history': history,
            'forecast': forecast_list,
            'indicators': indicators,
        }

        _set_cache(symbol, response_data)
        return jsonify(response_data)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to fetch data: {str(e)}'}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5003)
