import numpy as np
import pandas as pd


def extract_quarter_data(df):
    open_value = df.iloc[-1]["Open"]

    begin_close = df.iloc[0]["Close"]
    close_value = df.iloc[-1]["Close"]
    average_close = df["Close"].mean()

    volume = df.iloc[-1]['Volume']
    average_volume = df['Volume'].mean()

    high = df['High'].max()
    low = df['Low'].min()

    close_last_day = np.roll(df["Close"].to_numpy(), 1)[1:-1]
    close_daily = df["Close"].to_numpy()[1:-1]
    close_daily_performance = ((close_daily / close_last_day) - 1) * 100
    stock_series = pd.Series({'Company': pd.unique(df['Company'])[0], 'Last day Open': open_value,
                              'Begin day Close': begin_close, 'Last day Close': close_value,
                              'Average Price': average_close, 'Average Performance': np.mean(close_daily_performance),
                              'Volume': volume, 'Average Volume': average_volume,
                              'Quarter High': high, 'Quarter Low': low})
    return stock_series



def extract_stock(df):
    df['Quarter'] = df["Date"].dt.to_period("Q")
    df_2023Q1 = df[df['Quarter'] == '2023Q1']
    stocks_data = df_2023Q1.groupby(by=["Company"])[
        ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Company']].apply(extract_quarter_data)
    return stocks_data.reset_index(drop=True)


def main():
    stock_performance = pd.read_csv("../data/stocks_performance.csv")
    stock_info = pd.read_csv("../data/stocks_info.csv")

    stock_data = pd.read_csv("../data/stocks_data.csv", parse_dates=[0], date_format='%Y-%m-%d')
    stock_data_extract = extract_stock(stock_data)
    df_merge1 = pd.merge(stock_data_extract, stock_performance, on='Company')
    df_merge2 = pd.merge(df_merge1, stock_info, on='Company')
    df_merge2.to_csv("../data/stocks_data_processed.csv", index=False)


if __name__ == '__main__':
    main()
