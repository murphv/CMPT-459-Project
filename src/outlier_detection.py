import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import quantile_transform
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import numpy as np

from preprossessing_data import encode_category


def outlier_detection(df):
    stock = df['Stock']
    df_encoded = encode_category(df.copy())
    df_encoded = df_encoded.drop(columns=['Relative 2023Q2', 'Relative 2023Q2 Label', 'Stock'])
    lof = LocalOutlierFactor(n_neighbors=20)
    lof_outlier = lof.fit_predict(df_encoded)

    isf = IsolationForest(random_state=459)
    isf_outlier = isf.fit_predict(df_encoded)

    df_encoded_pca = PCA(n_components=2, random_state=459).fit_transform(df_encoded)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    colors = np.array(["#377eb8", "#ff7f00"])
    axs[0].scatter(df_encoded_pca[:, 0],
                   df_encoded_pca[:, 1], s=10, color=colors[(lof_outlier + 1) // 2])
    axs[0].set_title("Local Outlier Factor result")

    axs[1].scatter(df_encoded_pca[:, 0],
                   df_encoded_pca[:, 1], s=10, color=colors[(isf_outlier + 1) // 2])
    axs[1].set_title("Isolation Forest result")
    plt.savefig("../output/plot/outlier_result")

    plt.close(fig)

    lof_outlier_count = lof_outlier[lof_outlier == -1].size
    isf_outlier_count = isf_outlier[isf_outlier == -1].size
    both = np.logical_and(lof_outlier == -1, isf_outlier == -1)
    both_count = lof_outlier[both].size

    outlier_result = pd.DataFrame({'Stock': stock,
                                   'Local Outlier Factor': lof_outlier,
                                   'Isolation Forest': isf_outlier,
                                   'Is both': both})
    outlier_result.to_csv('../data/stock_outlier.csv', index=False)

    df_remove_outlier = df[np.logical_not(both)]
    df_remove_outlier.to_csv('../data/stock_outlier_removed.csv', index=False)

    with open('../output/outlier_result.txt', 'w') as f:
        f.write(f'Number of outliers detected using Local Outlier Factor:\t{lof_outlier_count}\n'
                f'Number of outliers detected using Isolation Forest:\t{isf_outlier_count}\n'
                f'Number of outliers detected using both methods:\t{both_count}')

def main():
    df = pd.read_csv('../data/stocks_data_normalized.csv')
    outlier_detection(df)

if __name__ == '__main__':
    main()