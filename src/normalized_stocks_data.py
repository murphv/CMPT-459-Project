import matplotlib.pyplot as plt
from sklearn.preprocessing import quantile_transform
import pandas as pd
import numpy as np


def lognorm(arr):
    return np.log1p(arr)

def normalization(df):
    df['Volume'] = lognorm(df['Volume'])
    df['fullTimeEmployees'] = lognorm(df['fullTimeEmployees'])
    df['Average Daily Volume'] = lognorm(df['Average Daily Volume'])
    df['Last Open'] = quantile_transform(df['Last Open'].to_frame(), n_quantiles=900, output_distribution="normal", copy=True).squeeze()
    df['Begin Close'] = quantile_transform(df['Begin Close'].to_frame(), n_quantiles=900, output_distribution="normal", copy=True).squeeze()
    df['Last Close'] = quantile_transform(df['Last Close'].to_frame(), n_quantiles=900, output_distribution="normal", copy=True).squeeze()
    df['Average Daily Price'] = quantile_transform(df['Average Daily Price'].to_frame(), n_quantiles=900, output_distribution="normal", copy=True).squeeze()
    df['Average Daily Performance'] = quantile_transform(df['Average Daily Performance'].to_frame(), n_quantiles=900, output_distribution="normal", copy=True).squeeze()
    df['High'] = quantile_transform(df['High'].to_frame(), n_quantiles=900, output_distribution="normal", copy=True).squeeze()
    df['Low'] = quantile_transform(df['Low'].to_frame(), n_quantiles=900, output_distribution="normal", copy=True).squeeze()
    df['Performance 2023Q1'] = quantile_transform(df['Performance 2023Q1'].to_frame(), n_quantiles=900, output_distribution="normal", copy=True).squeeze()

    return df

def plot_histogram(df):
    numerical_columns = ['Last Open', 'Begin Close', 'Last Close', 'Average Daily Price', 'Average Daily Performance',
                         'Volume', 'Average Daily Volume', 'High', 'Low', 'Performance 2023Q1', 'fullTimeEmployees']

    fig, axs = plt.subplots(4, 3, figsize=(15, 12))
    for column, ax in zip(numerical_columns, axs.ravel()):
        ax.hist(df[column], bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f'Distribution of {column}')
        ax.set_ylabel('count')
    fig.savefig('output/plot/overall_distribution')
    plt.close(fig)


def main():
    df = pd.read_csv('data/stocks_data_processed_imputed.csv')
    df = normalization(df)
    plot_histogram(df)
    df = df.drop(columns=['Volume']) # we drop Volume since it is somewhat redundant by other features and due to sparseness
    df.to_csv('data/stocks_data_normalized.csv', index=False)


if __name__ == '__main__':
    main()
