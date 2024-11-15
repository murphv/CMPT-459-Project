import pandas as pd
import yfinance as yf
import sys
import time
import requests as req



sec_headers = {
    'User-Agent': 'jj (jj@gmail.com)',
    'Accept-Encoding': 'gzip, deflate'
}


def download_ticker_cik_mapping() -> dict:
    """
    Downloads the SEC's company tickers JSON file and returns a dictionary mapping tickers to CIKs.
    """
    url = 'https://www.sec.gov/files/company_tickers.json'
    print(f"Downloading company tickers from {url}...")
    time.sleep(0.1)  # Rate limiting
    
    response = req.get(url, headers=sec_headers)
    if response.status_code != 200:
        print(f"Failed to download ticker data. Status code: {response.status_code}")
        return {}
    else:
        print("Successfully downloaded ticker data.")
    
    data = response.json()
    ticker_cik_mapping = {}
    for item in data.values():
        ticker = item['ticker']
        cik_str = str(item['cik_str']).zfill(10)
        ticker_cik_mapping[ticker.upper()] = cik_str
    
    return ticker_cik_mapping


def main():
    
    data = pd.DataFrame()
    
    start_date = '2023-01-01'
    end_date = '2023-06-30'

    tick_cik_map = download_ticker_cik_mapping()
    companies_count = 0
    tot_compapnies_count = len(tick_cik_map)
    current_progress = 0

    for tick, _ in tick_cik_map.items():

        companies_count+=1
        if int(100 * companies_count / tot_compapnies_count) > current_progress:
            current_progress += 1
            print(f"{current_progress}% completed")
        
        stock_data = yf.Ticker(tick).history(start=start_date, end=end_date)
        if stock_data.empty:
            # print(f"No data found for ticker {tick}. Removing from the list.")
            continue

        stock_data.reset_index(inplace=True)

        # Select and rename columns to match your desired format
        stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        stock_data['OpenInt'] = 0  # Open Interest is generally used for options/futures
        stock_data['Company'] = tick
        
        # Convert 'Date' to string format if necessary
        stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')
        
        data = pd.concat([data, stock_data], axis=0, ignore_index=True)
        
    data.to_csv('data/stocks_data.csv', index=False)


    # now we append the Index data (SP500 DowJones NasDaq)
    # Special index tickers
    index_tickers = ['^DJI', '^IXIC', '^GSPC']
    index_data =[]

    for tick in index_tickers:
        stock_data = yf.Ticker(tick).history(start=start_date, end=end_date)
        if stock_data.empty:
            # print(f"No data found for ticker {tick}. Removing from the list.")
            continue

        stock_data.reset_index(inplace=True)

        # Select and rename columns to match your desired format
        stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        stock_data['OpenInt'] = 0  # Open Interest is generally used for options/futures
        stock_data['Company'] = tick

        # Convert 'Date' to string format if necessary
        stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')

        index_data = pd.concat([index_data, stock_data], axis=0, ignore_index=True)

    data.to_csv('data/index_data.csv', index=False)

        

if __name__ == "__main__":
    main()