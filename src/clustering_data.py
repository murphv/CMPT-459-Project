import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from preprossessing_data import encode_category


def clustering_by_class(df):
    df_classes = df['Relative 2023Q2 Label']
    unique_classes = ['Exceptionally Bad Performance', 'Bad Performance', 'Neutral', 'Good Performance']
    df_encoded = encode_category(df.copy())
    df_encoded = df_encoded.drop(columns=['Relative 2023Q2', 'Relative 2023Q2 Label', 'Stock'])
    df_encoded_pca = PCA(n_components=2, random_state=459).fit_transform(df_encoded)

    fig, ax = plt.subplots(figsize=(10, 10))
    colours = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
    ax.set_title("Clustering result from label")
    fig.tight_layout()
    for (class_, colour) in zip(unique_classes, colours):
        df_class = df_encoded_pca[df_classes == class_]
        ax.scatter(df_class[:, 0], df_class[:, 1], s=10, c=colour, label=class_)
    plt.legend()
    plt.savefig('output/plot/cluster_label')
    plt.close(fig)


def main():
    df = pd.read_csv('data/stocks_data_normalized.csv')
    clustering_by_class(df)


if __name__ == '__main__':
    main()
