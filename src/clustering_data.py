import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

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


def clustering_DBSCAN(df):
    df_encoded = encode_category(df.copy())
    df_encoded = df_encoded.drop(columns=['Relative 2023Q2', 'Relative 2023Q2 Label', 'Stock'])
    df_encoded_pca = PCA(n_components=2, random_state=459).fit_transform(df_encoded)
    db = DBSCAN(eps=5.0, min_samples=8).fit(df_encoded)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.tight_layout(pad=2)
    ax.set_title("Clustering result from DBSCAN")

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = labels == k
        df_k = df_encoded_pca[class_member_mask & core_samples_mask]
        ax.plot(df_k[:, 0], df_k[:, 1],  "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=10)

        df_noise = df_encoded_pca[class_member_mask & ~core_samples_mask]
        ax.plot(df_noise[:, 0], df_noise[:, 1],  "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=6)
    fig.savefig('output/plot/cluster_DBSCAN')
    plt.close(fig)

    with open('output/DBSCAN_result.txt', 'w') as f:
        f.write(f'Estimated number of clusters: {n_clusters_}\n'
                f'Estimated number of noise points: {n_noise_}\n'
                f'Silhouette Coefficient: {metrics.silhouette_score(df_encoded, labels):.6f}\n'
                f'Calinski-Harabasz Index: {metrics.calinski_harabasz_score(df_encoded, labels):.6f}\n'
                f'Davies-Bouldin Index: {metrics.davies_bouldin_score(df_encoded, labels):.6f}')

def clustering_kmeans_plusplus(df):
    df_encoded = encode_category(df.copy())
    df_encoded = df_encoded.drop(columns=['Relative 2023Q2', 'Relative 2023Q2 Label', 'Stock'])
    df_encoded_pca = PCA(n_components=2, random_state=459).fit_transform(df_encoded)

    num_clusters = np.arange(2, 12)

    silhouette_scores = []

    for num_cluster in num_clusters:
        kmeans = KMeans(init="k-means++", n_clusters=num_cluster, random_state=459).fit(df_encoded)
        labels = kmeans.labels_
        silhouette_scores.append(metrics.silhouette_score(df_encoded, labels))

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.tight_layout(pad=2)
    ax.set_title("Silhouette Coefficient for K-Means using K clusters")
    ax.plot(num_clusters, silhouette_scores, 'o--')
    ax.set_xlabel('K-number clusters')
    ax.set_ylabel('Silhouette score')
    fig.savefig("output/plot/silhouette_kmeans")
    plt.close(fig)

    best_cluster = num_clusters[np.argmax(silhouette_scores)]
    kmeans = KMeans(init="k-means++", n_clusters=best_cluster, random_state=459).fit(df_encoded)
    labels = kmeans.labels_
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.tight_layout(pad=2)
    ax.set_title("Clustering result from K-Means with k-means++ initialization")

    for k, col in zip(unique_labels, colors):
        class_member_mask = labels == k
        df_k = df_encoded_pca[class_member_mask]
        ax.plot(df_k[:, 0], df_k[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=10)
    fig.savefig('output/plot/cluster_kmeans')
    plt.close(fig)

    with open('output/kmeans_result.txt', 'w') as f:
        f.write(f'Number of clusters with the best result: {best_cluster}\n'
                f'Silhouette Coefficient: {metrics.silhouette_score(df_encoded, labels):.6f}\n'
                f'Calinski-Harabasz Index: {metrics.calinski_harabasz_score(df_encoded, labels):.6f}\n'
                f'Davies-Bouldin Index: {metrics.davies_bouldin_score(df_encoded, labels):.6f}')

def main():
    df = pd.read_csv('data/stocks_data_normalized.csv')
    clustering_by_class(df)
    clustering_DBSCAN(df)
    clustering_kmeans_plusplus(df)


if __name__ == '__main__':
    main()
