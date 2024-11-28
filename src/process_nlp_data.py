from sklearn.decomposition import PCA, KernelPCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_nlp_pca(df: pd.DataFrame):
    nlp_data = df.drop(columns=['Company', 'Year', 'Quarter'])
    pca = PCA(random_state=459)
    pca.fit(nlp_data.to_numpy())
    explained_variances = pca.explained_variance_ratio_
    cdf_variances = np.cumsum(explained_variances)
    fig, ax = plt.subplots()
    er_80 = np.argmax(cdf_variances >= 0.8)+1
    er_90 = np.argmax(cdf_variances >= 0.9)+1
    er_95 = np.argmax(cdf_variances >= 0.95)+1
    ax.plot(np.arange(1, cdf_variances.size+1), cdf_variances)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax.grid(axis='both', which='major', linewidth=1)
    ax.grid(axis='both', which='minor', linewidth=0.2)
    ax.set_title("Cumulative Explained Variance Ratio by number of features using PCA")
    ax.set_ylabel('Cumulative Explained Variance Ratio')
    ax.set_xlabel('Number of Features')
    plt.savefig('output/plot/nlp_pca_variances')
    with open('output/pca_variance.txt', 'w') as f:
        f.write(f'The PCA explains 80% variances at {er_80} first features\n'
                f'The PCA explains 90% variances at {er_90} first features\n'
                f'The PCA explains 95% variances at {er_95} first features')

def apply_pca(df):
    nlp_data = df.drop(columns=['Company', 'Year', 'Quarter'])
    pca = PCA(n_components=42, random_state=459)
    columns_label = [f'BERT_PCA_{i}' for i in np.arange(0, 42)]
    nlp_reduced = pca.fit_transform(nlp_data.to_numpy())
    nlp_df = pd.DataFrame(nlp_reduced, columns=columns_label)
    nlp_df['Company'] = df['Company']
    return nlp_df


def main():
    nlp_df = pd.read_csv('data/nlp_features.csv')
    plot_nlp_pca(nlp_df)
    nlp_reduced = apply_pca(nlp_df)
    nlp_reduced.to_csv('data/nlp_features_reduced.csv', index=False)


if __name__ == '__main__':
    main()
