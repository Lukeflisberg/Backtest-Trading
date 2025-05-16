
import yfinance as yf
import pandas as pd

ticker = 'AGNG'
data = yf.download(ticker, interval='5m', start='2023-05-20', end='2023-05-26')
data = data.reset_index()

# Format the date/time
data['DATE'] = data['Datetime'].dt.strftime('%Y%m%d')
data['TIME'] = data['Datetime'].dt.strftime('%H%M%S')
data['TICKER'] = f"{ticker}.US"
data['PER'] = '5'
data['OPENINT'] = 0

# Reorder columns
out = data[['TICKER', 'PER', 'DATE', 'TIME', 'Open', 'High', 'Low', 'Close', 'Volume', 'OPENINT']]
out.to_csv('AGNG_formatted.csv', index=False, header=False)

