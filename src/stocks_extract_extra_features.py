import time

import pandas as pd
import yfinance as yf

def extract_stocks_info(stock):
    info = yf.Ticker(stock).info
    print(f'Extracting info from {stock}')
    return pd.Series(info)

def main():
    stocks_data = pd.read_csv("../data/stocks_performance.csv")
    companies_list = stocks_data['Company']
    company_info_list = []
    for company in companies_list:
        company_info = extract_stocks_info(company)
        company_info_list.append(company_info)
        time.sleep(0.1)  # Rate limiting
    company_info_df = pd.DataFrame(company_info_list)
    company_info_df = company_info_df[['zip', 'sector', 'fullTimeEmployees', 'city', 'phone', 'state', 'country',
                                       'industry', 'profitMargins', 'revenueGrowth', 'recommendationKey',
                                       'totalCashPerShare', 'exchange']]
    company_info_df['Company'] = companies_list
    company_info_df.to_csv("../data/stocks_info.csv", index=False)

if __name__ == '__main__':
    main()