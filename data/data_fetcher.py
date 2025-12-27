import requests
import pandas as pd

def fetch_stock_data(symbol, api_key):
    
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    time_series_key = 'Time Series (Daily)'
    if time_series_key not in data:
        raise ValueError(f"Error fetching data: {data.get('Error Message', 'Unknown error')}")
    
    time_series = data[time_series_key]
    
    records = []
    for date, values in time_series.items():
        records.append({
            'date': date,
            'open': float(values['1. open']),
            'high': float(values['2. high']),
            'low': float(values['3. low']),
            'close': float(values['4. close']),
            'volume': int(values['5. volume'])
        })
    
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    api_key = '1HQHBPL1U7I1NGI9'
    df = fetch_stock_data('AAPL', api_key)
    print(df)
