from textwrap import wrap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from preprossessing_data import encode_category


def plot_histogram(df):
    numerical_columns = ['Last Open', 'Begin Close', 'Last Close', 'Average Daily Price', 'Average Daily Performance',
                         'Volume', 'Average Daily Volume', 'High', 'Low', 'Performance 2023Q1', 'fullTimeEmployees',
                         'BERT_PCA_0', 'BERT_PCA_1', 'BERT_PCA_2', 'BERT_PCA_40', 'BERT_PCA_41']

    fig, axs = plt.subplots(4, 4, figsize=(15, 12))
    fig.tight_layout(h_pad=2)
    for column, ax in zip(numerical_columns, axs.ravel()):
        ax.hist(df[column], bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f'Distribution of {column}', size=13)
        ax.set_ylabel('count')
    fig.savefig('output/plot/overall_distribution')
    plt.close(fig)
def plot_heatmap(df):
    df_encoded = encode_category(df.copy())
    df_encoded = df_encoded.drop(columns=['Relative 2023Q2 Label', 'Relative 2023Q2'])
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df_encoded.corr(numeric_only=True), cmap="YlGnBu")
    ax.set_title("Correlations Heatmap between features")
    fig.savefig("output/plot/heatmap")
    plt.close(fig)

def plot_cat_histogram(df):
    fig, axd = plt.subplot_mosaic([['sector', 'sector', 'sector'],
                                   ['exchange', 'exchange', 'exchange'],
                                   ['recommendationKey', 'country', 'Relative 2023Q2 Label']],
                                  figsize=(15, 12))
    fig.tight_layout(pad=2)
    for k, ax in axd.items():
        ax.set_title(f'Distribution of {k}')
        count = df[k].value_counts()
        ax.bar(count.index, count.array)
        ax.set_ylabel('Count')

        x_tick = ax.get_xticklabels()
        x_tick = [label.get_text().replace(' ', '\n') for label in x_tick ]
        print(x_tick)
        ax.set_xticklabels(x_tick)
    plt.savefig('output/plot/cat_distribution')
    plt.close(fig)


def main():
    df_before_normalized = pd.read_csv('data/stocks_data_processed_imputed.csv')
    df_post_normalize = pd.read_csv('data/stocks_data_normalized_v1.csv')

    plot_cat_histogram(df_post_normalize)
    plot_heatmap(df_before_normalized)

    df_post_normalize = df_post_normalize.drop(
        columns=['Volume'])  # we drop Volume since it is somewhat redundant by other features and due to sparseness
    df_post_normalize.to_csv('data/stocks_data_normalized.csv', index=False)


if __name__ == '__main__':
    main()