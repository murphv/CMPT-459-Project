import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, RFE, mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import os


def evaluate_model(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Accuracy: {mean_score:.4f} ± {std_score:.4f}")
    return mean_score, std_score


def main():
    desired_feature_sizes = [2, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70]
    feature_set_idx = 0
    feature_sets = []

    data = pd.read_csv('data/stocks_data_normalized.csv')

    # One hot encoding the categorical features
    categorical_features = ['sector', 'country', 'recommendationKey', 'exchange']
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

    data.to_csv('data/normalized_encoded_data.csv', index=False)


    ## Feature Analysis and Selection
    X = data.copy()
    X = X.drop(columns=['Relative 2023Q2', 'Relative 2023Q2 Label', 'Stock'])
    y_class = data['Relative 2023Q2 Label']
    y_reg = data['Relative 2023Q2']

    # i) [Variance Thresholding] getting high and medium variance features
    features_all = X.columns
    feature_sets.append({'Method': 'All Features', 'Parameter': None, 'NumFeatures': len(features_all), 'Features': features_all, 'Idx': feature_set_idx})
    feature_set_idx += 1

    sel = VarianceThreshold(threshold=0.3)
    sel.fit(X)
    features_mid_variance = X.columns[sel.get_support()]
    feature_sets.append({'Method': 'VarianceThreshold', 'Parameter': 0.3, 'NumFeatures': len(features_mid_variance), 'Features': features_mid_variance, 'Idx': feature_set_idx})
    feature_set_idx += 1

    sel = VarianceThreshold(threshold=0.9)
    sel.fit(X)
    features_high_variance = X.columns[sel.get_support()]
    feature_sets.append({'Method': 'VarianceThreshold', 'Parameter': 0.9, 'NumFeatures': len(features_high_variance), 'Features': features_high_variance , 'Idx': feature_set_idx})
    feature_set_idx += 1

    # ii) [Mutual Information] for both classification and regression labels
    mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mutual_info_classif(X, y_class)})
    mi_cls_df = mi_scores_df.sort_values('MI Score', ascending=False)

    for size in desired_feature_sizes:
        features = mi_cls_df['Feature'].head(size).to_numpy()
        feature_sets.append({'Method': 'MutualInfo_Classification', 'Parameter': size, 'NumFeatures': len(features), 'Features': features, 'Idx': feature_set_idx})
        feature_set_idx += 1

    mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mutual_info_regression(X, y_reg)})
    mi_reg_df = mi_scores_df.sort_values('MI Score', ascending=False)

    for size in desired_feature_sizes:
        features = mi_reg_df['Feature'].head(size).to_numpy()
        feature_sets.append({'Method': 'MutualInfo_Regression', 'Parameter': size, 'NumFeatures': len(features), 'Features': features, 'Idx': feature_set_idx})
        feature_set_idx += 1

    # iii) [Recursive Feature Elimination] using logistic regression model
    model = LogisticRegression(max_iter=1000)
    for size in desired_feature_sizes:
        features = X.columns[RFE(estimator=model, n_features_to_select=size, step=1).fit(X, y_class).support_]
        feature_sets.append({'Method': 'RFE', 'Parameter': size, 'NumFeatures': len(features), 'Features': features, 'Idx': feature_set_idx})
        feature_set_idx += 1

    # Evaluate model and collect results based on each of feature_sets
    results = []

    for fs in feature_sets:
        method = fs['Method']
        param = fs['Parameter']
        num_features = fs['NumFeatures']
        features = fs['Features']
        ids = fs['Idx']
        mean_score, std_score = evaluate_model(model, X[features], y_class)
        results.append({'Method': method, 'Parameter': param, 'NumFeatures': num_features, 'Accuracy': mean_score, 'Std': std_score, 'Idx': ids})

    results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    results_df.to_csv('data/feature_selector_results.csv', index=False)

    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=results_df, x='NumFeatures', y='Accuracy', hue='Method', marker='o')
    plt.title('Model Accuracy vs Number of Features for Different Feature Selection Methods')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.legend(title='Method')
    plt.savefig('output/plot/feature_selection_methods.png')

    # Saving the best features result for classification / Regression purposes
    best_feature_idx = results_df.iloc[0]['Idx']
    best_feature_set = next((fs for fs in feature_sets if fs['Idx'] == best_feature_idx), None)
    best_features_df = pd.DataFrame(best_feature_set['Features'], columns=['Feature'])
    best_features_df.to_csv('data/best_features_classification.csv', index=False)

    for i in range(0,results_df.shape[0]):
        if results_df.iloc[i]['Method'] == 'MutualInfo_Regression':
            best_feature_idx = results_df.iloc[i]['Idx']
            break
    
    best_feature_set = next((fs for fs in feature_sets if fs['Idx'] == best_feature_idx), None)
    best_features_df = pd.DataFrame(best_feature_set['Features'], columns=['Feature'])
    best_features_df.to_csv('data/best_features_regression.csv', index=False)
    


if __name__ == "__main__":
    main()
