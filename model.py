import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


FEATURE_COLS = [
    'sma_5_ratio', 'sma_20_ratio', 'rsi', 'momentum_5',
    'volume_ratio', 'volatility', 'hl_ratio',
]


def _compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _build_features(df):
    d = df.copy()
    sma_5 = d['close'].rolling(5).mean()
    sma_20 = d['close'].rolling(20).mean()

    d['sma_5_ratio'] = d['close'] / sma_5
    d['sma_20_ratio'] = d['close'] / sma_20
    d['rsi'] = _compute_rsi(d['close'])
    d['momentum_5'] = d['close'].pct_change(5)
    d['volume_ratio'] = d['volume'] / d['volume'].rolling(20).mean()
    d['volatility'] = d['close'].rolling(10).std() / d['close']
    d['hl_ratio'] = (d['high'] - d['low']) / d['close']

    # Target: next-day return (more stationary than absolute price)
    d['target'] = d['close'].shift(-1) / d['close'] - 1

    d = d.dropna(subset=FEATURE_COLS)
    return d


def _make_model():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )


def predict_price(historical_data):
    """Return dict with point prediction, confidence range, and accuracy metrics."""
    last_close = float(historical_data['close'].iloc[-1])

    if len(historical_data) < 30:
        return _fallback(last_close)

    data = _build_features(historical_data)
    train = data.dropna(subset=['target'])

    if len(train) < 10:
        return _fallback(last_close)

    X_train = train[FEATURE_COLS].values
    y_train = train['target'].values
    X_pred = data[FEATURE_COLS].iloc[-1:].values
    current = float(data['close'].iloc[-1])

    model = _make_model()
    model.fit(X_train, y_train)

    pred_ret = model.predict(X_pred)[0]
    prediction = current * (1 + pred_ret)

    tree_rets = np.array([t.predict(X_pred)[0] for t in model.estimators_])
    std = tree_rets.std()
    low = current * (1 + pred_ret - 1.5 * std)
    high = current * (1 + pred_ret + 1.5 * std)

    mae_pct, direction_acc = _evaluate(train)

    return {
        'prediction': float(prediction),
        'low': float(low),
        'high': float(high),
        'mae_percent': mae_pct,
        'direction_accuracy': direction_acc,
    }


def forecast_prices(historical_data, days=5):
    """Multi-horizon forecast: train a separate model per horizon (1..N days)."""
    last_close = float(historical_data['close'].iloc[-1])

    if len(historical_data) < 30:
        return [_forecast_fallback(last_close, d) for d in range(1, days + 1)]

    data = _build_features(historical_data)
    if len(data) < 10:
        return [_forecast_fallback(last_close, d) for d in range(1, days + 1)]

    X_pred = data[FEATURE_COLS].iloc[-1:].values
    current = float(data['close'].iloc[-1])

    results = []
    for day in range(1, days + 1):
        target_n = data['close'].shift(-day) / data['close'] - 1
        valid_mask = target_n.notna()
        train_idx = data.index[valid_mask]

        if len(train_idx) < 10:
            results.append(_forecast_fallback(last_close, day))
            continue

        X = data.loc[train_idx, FEATURE_COLS].values
        y = target_n.loc[train_idx].values

        model = _make_model()
        model.fit(X, y)

        pred_ret = model.predict(X_pred)[0]
        price = current * (1 + pred_ret)

        tree_rets = np.array([t.predict(X_pred)[0] for t in model.estimators_])
        std = tree_rets.std()

        results.append({
            'day': day,
            'prediction': float(price),
            'low': float(current * (1 + pred_ret - 1.5 * std)),
            'high': float(current * (1 + pred_ret + 1.5 * std)),
        })

    return results


def compute_indicators(df):
    """Compute technical indicators for chart overlays."""
    sma_5 = df['close'].rolling(5).mean()
    sma_20 = df['close'].rolling(20).mean()
    rsi = _compute_rsi(df['close'])

    indicators = {'sma_5': [], 'sma_20': [], 'rsi': []}

    for i in range(len(df)):
        date_str = df.iloc[i]['date'].strftime('%Y-%m-%d')
        if pd.notna(sma_5.iloc[i]):
            indicators['sma_5'].append({
                'time': date_str,
                'value': round(float(sma_5.iloc[i]), 2),
            })
        if pd.notna(sma_20.iloc[i]):
            indicators['sma_20'].append({
                'time': date_str,
                'value': round(float(sma_20.iloc[i]), 2),
            })
        if pd.notna(rsi.iloc[i]):
            indicators['rsi'].append({
                'time': date_str,
                'value': round(float(rsi.iloc[i]), 2),
            })

    return indicators


def _evaluate(train):
    if len(train) <= 20:
        return None, None

    n = min(10, len(train) // 4)
    tr, te = train.iloc[:-n], train.iloc[-n:]

    ev = _make_model()
    ev.fit(tr[FEATURE_COLS].values, tr['target'].values)
    preds = ev.predict(te[FEATURE_COLS].values)

    actual_px = te['close'].values * (1 + te['target'].values)
    pred_px = te['close'].values * (1 + preds)
    mae = mean_absolute_error(actual_px, pred_px)
    mae_pct = round((mae / actual_px.mean()) * 100, 2)

    a_dir = np.sign(te['target'].values)
    p_dir = np.sign(preds)
    direction_acc = round(float((a_dir == p_dir).mean()) * 100, 1)

    return mae_pct, direction_acc


def _fallback(last_close):
    return {
        'prediction': last_close,
        'low': last_close * 0.98,
        'high': last_close * 1.02,
        'mae_percent': None,
        'direction_accuracy': None,
    }


def _forecast_fallback(last_close, day):
    return {
        'day': day,
        'prediction': last_close,
        'low': last_close * 0.98,
        'high': last_close * 1.02,
    }


if __name__ == "__main__":
    from data.data_fetcher import fetch_stock_data

    df = fetch_stock_data('AAPL')
    result = predict_price(df)
    forecasts = forecast_prices(df)

    print(f"Latest close:       ${df['close'].iloc[-1]:.2f}")
    print(f"Predicted next day: ${result['prediction']:.2f}")
    print(f"Range:              ${result['low']:.2f} – ${result['high']:.2f}")
    if result['mae_percent'] is not None:
        print(f"MAE:                {result['mae_percent']}%")
    if result['direction_accuracy'] is not None:
        print(f"Direction accuracy: {result['direction_accuracy']}%")
    print(f"\n5-day forecast:")
    for f in forecasts:
        print(f"  Day {f['day']}: ${f['prediction']:.2f} (${f['low']:.2f} – ${f['high']:.2f})")
