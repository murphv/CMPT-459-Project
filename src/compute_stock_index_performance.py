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
    quarter_performance = df.groupby(by=["Quarter"])[['Quarter', 'Close', 'Company']].apply(extract_quarter)
    company_name = pd.unique(df["Company"])[0]
    performance = [np.nan, np.nan]
    if '2023Q1' in quarter_performance.index:
        performance[0] = quarter_performance.loc['2023Q1']
    if '2023Q2' in quarter_performance.index:
        performance[1] = quarter_performance.loc['2023Q2']
    performance_series = pd.Series({"Company": company_name,
                                    "Performance 2023Q1": performance[0],
                                    "Performance 2023Q2": performance[1]})
    return performance_series


def extract_stock_performance(df):
    df['Quarter'] = df["Date"].dt.to_period("Q")
    stock_performance = df.groupby(by=["Company"])[['Quarter', 'Close', 'Company']].apply(extract_quarter_performance)
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

    stocks_performance["Relative 2023Q1"] = stocks_performance['Performance 2023Q1'] - snp_index['Performance 2023Q1']
    stocks_performance["Relative 2023Q2"] = stocks_performance['Performance 2023Q2'] - snp_index['Performance 2023Q2']
    stocks_performance.to_csv("../data/stocks_performance.csv", index=False)


if __name__ == '__main__':
    main()
