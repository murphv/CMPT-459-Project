import datetime
import glob
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

DATETIME_FORMAT = '%Y-%m-%d'
RANDOM_SEED = 459


def list_stocks(input_dir):
    dir_url = Path(input_dir)
    stock_name = [file.stem for file in dir_url.glob('*.txt')]
    return stock_name


def extract_beginning_date(stocks_names):
    stocks_date = []
    for stock_name in stocks_names:
        with open(f'Stocks/{stock_name}.txt') as file:
            second_line = file.readlines()[1]
            date_str = second_line.split(',', 1)[0]
            date = datetime.strptime(date_str, DATETIME_FORMAT)
            stocks_date.append(date)
    return stocks_date


def sampling_stock_by_years(stocks_df, year_segments, num_samples):
    final_samples = []
    rng = np.random.RandomState(seed=RANDOM_SEED)
    for segment, num in zip(year_segments, num_samples):
        stocks_stratum = stocks_df[stocks_df['Beginning date'] >=
                                   datetime.strptime(f'{segment[0]}-01-01', DATETIME_FORMAT)]
        stocks_stratum = stocks_stratum[stocks_stratum['Beginning date'] <
                                        datetime.strptime(f'{segment[1]}-01-01', DATETIME_FORMAT)]
        stocks_samples = stocks_stratum.sample(n=num, random_state=rng)
        final_samples.extend(stocks_samples['Stock name'])
    return final_samples


def main():
    stock_names = list_stocks('Stocks/')
    # remove S&P as we use it as main indicator
    stock_names.remove('snp.us')
    # filter stock with empty file size
    stocks_date = extract_beginning_date(stock_names)
    stocks_df = pd.DataFrame({'Stock name': stock_names, 'Beginning date': stocks_date})
    # Filter out stock which have recorded date later than 2016
    stocks_df = stocks_df[stocks_df['Beginning date'] <= datetime.strptime('2016-01-01', DATETIME_FORMAT)]
    samples = sampling_stock_by_years(stocks_df, [[1960, 1980], [1980, 2000], [2000, 2010], [2010, 2020]],
                                      [10, 40, 150, 100])
    samples_df = pd.DataFrame({'stocks_name': samples})
    samples_df['stocks_name'] = samples_df['stocks_name'].str.split('.', n=1).str[0]
    samples_df.to_csv('output/samples_stocks.csv', index=False)


if __name__ == '__main__':
    main()
