from datetime import datetime
import numpy as np
import pandas as pd

from src.compute_stock_performance import filter_company_with_no_report


def extract_quarter(df):
    open_value = df.iloc[0]["Close"]
    close_value = df.iloc[-1]["Close"]
    peformance_percentage = ((close_value / open_value) - 1) * 100.0
    return peformance_percentage


def extract_quarter_performance(df):
    average_monthly = df.drop(columns=['Company']).groupby(by=['Month']).mean().reset_index()
    company_start_Q2 = average_monthly[average_monthly['Month'] <= 3].drop(columns='Month').mean()['Close']
    company_end_Q2 = average_monthly[average_monthly['Month'] == 6].drop(columns='Month').mean()['Close']

    # Compute the stock change for 2023Q3
    stock_change_Q2 = ((company_end_Q2 - company_start_Q2) / company_start_Q2) * 100

    company_start_Q1 = average_monthly[average_monthly['Month'] <= 1].drop(columns='Month').mean()['Close']
    company_end_Q1 = average_monthly[average_monthly['Month'] == 3].drop(columns='Month').mean()['Close']
    stock_change_Q1 = ((company_end_Q1 - company_start_Q1) / company_start_Q1) * 100
    performance_series = pd.Series({'Company': df['Company'].unique()[0], 'Performance 2023Q1': stock_change_Q1,
                                    'Performance 2023Q2': stock_change_Q2})

    return performance_series


def extract_stock_performance(df):
    df['Month'] = df["Date"].dt.month
    stock_performance = df.groupby(by=["Company"])[['Month', 'Close', 'Company']].apply(extract_quarter_performance)
    return stock_performance


def main():
    index_data = pd.read_csv("../data/index_data.csv", parse_dates=[0], date_format='%Y-%m-%d')
    index_performance = extract_stock_performance(index_data)
    index_performance.to_csv("../data/index_performance.csv", index=False)
    snp_index = index_performance.loc["^GSPC"]
    stocks_data = pd.read_csv("../data/stocks_data.csv", parse_dates=[0], date_format='%Y-%m-%d')
    stocks_performance = extract_stock_performance(stocks_data)
    companies_list = filter_company_with_no_report(stocks_performance['Company'].unique())
    stocks_performance = stocks_performance[stocks_performance['Company'].isin(companies_list)]

    stocks_performance["Relative 2023Q2"] = stocks_performance['Performance 2023Q2'] - snp_index['Performance 2023Q2']
    stocks_performance.to_csv("../data/stocks_performance.csv", index=False)


if __name__ == '__main__':
    main()
