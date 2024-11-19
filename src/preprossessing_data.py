from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def plot_missing_data(df: pd.DataFrame, title):
    nlp_columns = [f'BERT_PCA_{i}' for i in np.arange(0, 42)]
    non_nlp_df = df.drop(columns=nlp_columns)
    missing_data = non_nlp_df.isna().sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.barh(missing_data.index, missing_data.values, color='maroon')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(axis='x', which='major', linewidth=1)
    ax.grid(axis='x', which='minor', linewidth=0.2)
    ax.set_xlabel('Features')
    ax.set_ylabel('No. missing data')
    ax.set_title('Plot of number missing data')
    plt.savefig(f'../output/plot/{title}')
    plt.close(fig)


def drop_or_process_missing_value(df):
    # drop columns with too many missing value
    df = df.drop(columns=['profitMargins', 'revenueGrowth', 'totalCashPerShare'])
    # drop unused columns
    df = df.drop(columns=['Performance 2023Q2', 'phone', 'Company', 'state', 'city', 'zip', 'industry'])

    # remove samples in important features
    df = df[df['Performance 2023Q1'].notna()]
    df = df[df['country'].notna()]
    df = df[df['sector'].notna()]

    # impute value
    df['recommendationKey'] = df['recommendationKey'].fillna('none')

    # Change category
    df['country'] = simplify_country(df)

    df_cat = encode_category(df.copy())
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    imputer_result = imputer.fit_transform(df_cat)
    df['fullTimeEmployees'] = imputer_result[:, 12].astype(np.int64)
    # df['fullTimeEmployees'] = df['fullTimeEmployees'].fillna(df['fullTimeEmployees'].median())

    return df


def encode_category(df):
    sector_encoding = {
        'Technology': 0,
        'Healthcare': 1,
        'Industrials': 2,
        'Consumer Cyclical': 3,
        'Financial Services': 4,
        'Consumer Defensive': 5,
        'Communication Services': 6,
        'Basic Materials': 7,
        'Energy': 8,
        'Real Estate': 9,
        'Utilities': 10
    }
    df['sector'] = df['sector'].map(sector_encoding)

    recommendation_encoding = {
        'none': 0,
        'hold': 1,
        'buy': 2,
        'strong_buy': 3,
        'underperform': -1
    }
    df['recommendationKey'] = df['recommendationKey'].map(recommendation_encoding)

    # Due to the number of countries, we categorize them into US, America, Europe, Asia and Pacific, Africa
    country_encoding = {
        'United States': 0,
        'North America': 1,
        'Asia and Pacific': 2,
        'Europe': 3,
        'Africa': 4
    }
    df['country'] = df['country'].map(country_encoding)

    exchange_encoding = {
        'NMS': 0,
        'NYQ': 1,
        'PNK': 2,
        'NCM': 3,
        'OQB': 4,
        'OEM': 5,
        'NGM': 6,
        'ASE': 7,
        'OQX': 8
    }
    df['exchange'] = df['exchange'].map(exchange_encoding)
    return df


def process_category(df):
    with open('../output/category_counts.txt', 'w') as f:
        f.write(df['Relative 2023Q2 Label'].value_counts().to_string())
        f.write('\n\n')
        f.write(df['sector'].value_counts().to_string())
        f.write('\n\n')
        f.write(df['recommendationKey'].value_counts().to_string())
        f.write('\n\n')
        f.write(df['country'].value_counts().to_string())
        f.write('\n\n')
        f.write(df['exchange'].value_counts().to_string())

    return df


def simplify_country(df):
    country_encoding = {
        'United States': 'United States',
        'Canada': 'North America',
        'China': 'Asia and Pacific',
        'Hong Kong': 'Asia and Pacific',
        'Ireland': 'Europe',
        'United Kingdom': 'Europe',
        'Japan': 'Asia and Pacific',
        'Switzerland': 'Europe',
        'Malaysia': 'Asia and Pacific',
        'Singapore': 'Asia and Pacific',
        'Taiwan': 'Asia and Pacific',
        'Cayman Islands': 'North America',
        'Australia': 'Asia and Pacific',
        'Malta': 'Europe',
        'Mexico': 'North America',
        'Israel': 'Europe',
        'Netherlands': 'Europe',
        'South Africa': 'Africa',
        'Kazakhstan': 'Asia and Pacific',
        'Kenya': 'Africa'
    }
    return df['country'].map(country_encoding)


def create_label(df):
    df['Relative 2023Q2 Label'] = pd.cut(x=df['Relative 2023Q2'],
                                         bins=[df['Relative 2023Q2'].min() - 1, -50, -10, 10,
                                               df['Relative 2023Q2'].max() + 1],
                                         labels=['Exceptionally Bad Performance', 'Bad Performance', 'Neutral',
                                                 'Good Performance'])
    return df


def plot_outcome(df):
    outcome = df['Relative 2023Q2']
    outcome = outcome[
        outcome <= 200]  # Since there are a number of sample achieve >200%, we saw filter them for plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(outcome, bins=50, kde=True, color='lightgreen', edgecolor='red')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(axis='both', which='major', linewidth=0.5)
    ax.set_title('Histogram of Stock performance in 2023Q2 relative to index stock performance')
    ax.set_xlabel('Relative percentage')
    plt.savefig('../output/plot/histogram_relative')
    plt.close(fig)


def plot_employees(df):
    employees_num = df['fullTimeEmployees']
    # employees_num = np.log10(employees_num)
    fig, ax = plt.subplots(figsize=(10, 5))
    logbins = np.logspace(np.log10(employees_num.min()), np.log10(employees_num.max()), 30)
    sns.histplot(employees_num, bins=logbins, color='lightgreen', edgecolor='red')
    ax.grid(axis='both', which='major', linewidth=0.5)
    plt.xscale('log')
    ax.set_title('Histogram of Employees employed full-time')
    ax.set_xlabel('Full-time Employees')
    plt.savefig('../output/plot/histogram_employee')
    plt.close(fig)


def plot_heatmap(df):
    df_encoded = encode_category(df.copy())
    df_encoded = df_encoded.drop(columns=['Relative 2023Q2 Label', 'Relative 2023Q2'])
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df_encoded.corr(numeric_only=True), cmap="YlGnBu")
    ax.set_title("Correlations Heatmap between features")
    fig.savefig("../output/plot/heatmap")
    plt.close(fig)

def main():
    df = pd.read_csv('../data/stocks_data_processed.csv')
    stocks = df['Company']

    plot_missing_data(df, 'missing_data')
    df_process = drop_or_process_missing_value(df)
    plot_outcome(df_process)
    plot_employees(df_process)
    df_process = create_label(df_process)
    df_process = process_category(df_process)
    df_process['Stock'] = stocks #Rename and attach stock name at the end

    plot_missing_data(df_process, 'missing_data_post')
    plot_heatmap(df_process)
    df_process.to_csv('../data/stocks_data_processed_imputed.csv', index=False)


if __name__ == '__main__':
    main()
