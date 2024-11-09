import requests as req
import os
import time
import sys
from datetime import datetime
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import yfinance as yf


log = ''

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


def get_filings_by_company(cik: str, ticker: str, retries=3) -> dict:
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    for attempt in range(retries):
        time.sleep(0.1)

        try:
            response = req.get(submissions_url, headers=sec_headers)
            if response.status_code == 200:
                print(f"Successfully fetched submissions for {ticker}.")
                submissions = response.json()

                # Check if 'filings' and 'recent' keys are present
                if 'filings' in submissions and 'recent' in submissions['filings']:
                    return submissions['filings']['recent']
                else:
                    print(f"'filings' or 'recent' not found in submissions for {ticker}")
                    return None
            else:
                print(f"Failed to fetch submissions for {ticker}. Status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred while fetching submissions for {ticker}: {e}")

        # Wait before retrying
        time.sleep(1)

    # After retries
    print(f"All retries failed for {ticker}")
    return None



def get_q_report(cik: str, year: int, q: int, filings: dict) -> str:
    """
    Extracts the 10Q report of a company
    """

    filing_dates = filings['filingDate']
    accession_numbers = filings['accessionNumber']
    primary_documents = filings['primaryDocument']
    
    # Find indices where form is '10-Q'
    indices_10q = [i for i, x in enumerate(filings['form']) if x == '10-Q']

    for idx in indices_10q:
        date_str = filing_dates[idx]
        date = datetime.strptime(date_str, '%Y-%m-%d')
        start_month = (q - 1) * 3 + 1
        end_month = q * 3

        if datetime(year, start_month, 1) <= date <= datetime(year, end_month, 31):
            print(f"\nFound 10-Q filing at index {idx}:")
            
            # Construct the URL for the filing
            accession_number = accession_numbers[idx].replace('-', '')
            primary_document = primary_documents[idx]
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/{primary_document}"

            return filing_url

    print(f"No 10Q files matched for quarter {q} of {year}")
    return None


def parse_html_to_text(html_content: str) -> str:
    """
    Parses the content of an HTML string
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    text = soup.get_text(separator=' ')

    # cleaning up the text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    text = '\n'.join(text.split('\n')[6:])
    return text


def sanitize_ticker(ticker: str) -> str:
    """
    Sanitizes the ticker symbol by removing any characters starting from an underscore or dash.
    Example: 'BAC_D' -> 'BAC', 'COF-WS' -> 'COF'
    """
    ticker = ticker.strip().upper()
    # Split the ticker at '_' or '-' and take the first part
    ticker = ticker.split('_')[0].split('-')[0]
    return ticker
            

def get_sanitized_tickers(df, ciks):
    # Convert to uppercase, strip whitespace, and drop NaN values
    tickers = pd.Series(df).str.upper().str.strip().dropna().to_numpy()
    print("Tickers to process:")
    
    valid_tickers = []
    invalid_tickers = []

    for og_ticker in tickers:
        new_ticker = sanitize_ticker(og_ticker)
        cik = ciks.get(new_ticker)
        if cik:
            valid_tickers.append((new_ticker, cik))
        else:
            invalid_tickers.append(og_ticker)
            print(f"Invalid or unrecognized ticker after sanitization: {og_ticker} -> {new_ticker}")

    pd.DataFrame(valid_tickers, columns=['ticker', 'cik']).to_csv('valid_tickers.csv')
    return valid_tickers


def main():
    path_dir = '../data/SEC_Filings'
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    year = int(sys.argv[1])
    quarter = int(sys.argv[2])

    log = ''
    
    # Download the mapping of tickers to CIKs
    company_ciks = download_ticker_cik_mapping()

    # Read the ticker csv file
    data = pd.read_csv('../data/stocks_data.csv')
    companies_df = data['Company'].unique()
    stocks_data = get_sanitized_tickers(companies_df, company_ciks)
    

    for ticker, cik in stocks_data:

        # Getting the Recent filings dictionary for the company
        filings_recent = get_filings_by_company(cik, ticker)
        if not filings_recent:
            log += f"No recent filings found for ticker: {ticker}\n"
            continue  

        # Getting the url of the 10q report corresponding to the year and quarter
        filing_url = get_q_report(cik, year, quarter, filings_recent)
        if not filing_url:
            log += f"No 10-Q filing found for {ticker} in Q{quarter} {year}\n"
            continue

        log_record = f"{ticker} in Q{quarter} {year}"
        
        log += f"\nAttempting to download the filing for {log_record} \n"

        time.sleep(0.15)  # Rate limiting  
        try:
            filing_response = req.get(filing_url, headers=sec_headers)
        except Exception as e:
            log += f"An error occurred while downloading the filing for {log_record}: {e}\n"
            continue

        if filing_response.status_code == 200:
            log += f"\nSuccessfully dowloaded the 10q for {log_record} \n"

            text = parse_html_to_text(filing_response.content)

            # Save the Filing text
            company_dir = os.path.join(path_dir, ticker)
            if not os.path.exists(company_dir):
                os.makedirs(company_dir)

            file_path = os.path.join(company_dir, f"{ticker}_10-Q_{year}_{quarter}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)

        else:
            log += f"Failed to download the filing for {ticker}. Status code: {filing_response.status_code}\n"

    # Write the log
    if log:
        with open('../output/extract_10q_log.txt', 'w', encoding='utf-8') as f:
            f.write(log)

    

if __name__ == "__main__":
    main()
