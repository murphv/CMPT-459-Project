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



def evaluate_model(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Accuracy: {mean_score:.4f} ± {std_score:.4f}")
    return mean_score, std_score


def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')  # Adjust average if multiclass
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))


def main():
    desired_feature_sizes = [2, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70]
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
    feature_sets.append({'Method': 'All Features', 'Parameter': None, 'NumFeatures': len(features_all), 'Features': features_all})

    sel = VarianceThreshold(threshold=0.3)
    sel.fit(X)
    features_mid_variance = X.columns[sel.get_support()]
    feature_sets.append({'Method': 'VarianceThreshold', 'Parameter': 0.3, 'NumFeatures': len(features_mid_variance), 'Features': features_mid_variance})

    sel = VarianceThreshold(threshold=0.9)
    sel.fit(X)
    features_high_variance = X.columns[sel.get_support()]
    feature_sets.append({'Method': 'VarianceThreshold', 'Parameter': 0.9, 'NumFeatures': len(features_high_variance), 'Features': features_high_variance})

    # ii) [Mutual Information] for both classification and regression labels
    mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mutual_info_classif(X, y_class)})
    mi_cls_df = mi_scores_df.sort_values('MI Score', ascending=False)
    mi_cls_df.to_csv('data/mutual_information_classification.csv', index=False)

    for size in desired_feature_sizes:
        features = mi_cls_df['Feature'].head(size).to_numpy()
        feature_sets.append({'Method': 'MutualInfo_Classification', 'Parameter': size, 'NumFeatures': len(features), 'Features': features})

    mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mutual_info_regression(X, y_reg)})
    mi_reg_df = mi_scores_df.sort_values('MI Score', ascending=False)
    mi_reg_df.to_csv('data/mutual_information_regression.csv', index=False)

    for size in desired_feature_sizes:
        features = mi_reg_df['Feature'].head(size).to_numpy()
        feature_sets.append({'Method': 'MutualInfo_Regression', 'Parameter': size, 'NumFeatures': len(features), 'Features': features})

    # iii) [Recursive Feature Elimination] using logistic regression model
    model = LogisticRegression(max_iter=1000)
    for size in desired_feature_sizes:
        features = X.columns[RFE(estimator=model, n_features_to_select=size, step=1).fit(X, y_class).support_]
        feature_sets.append({'Method': 'RFE', 'Parameter': size, 'NumFeatures': len(features), 'Features': features})

    # Evaluate model and collect results based on each of feature_sets
    results = []

    for fs in feature_sets:
        method = fs['Method']
        param = fs['Parameter']
        num_features = fs['NumFeatures']
        features = fs['Features']
        mean_score, std_score = evaluate_model(model, X[features], y_class)
        results.append({'Method': method, 'Parameter': param, 'NumFeatures': num_features, 'Accuracy': mean_score, 'Std': std_score})

    results_df = pd.DataFrame(results)

    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=results_df, x='NumFeatures', y='Accuracy', hue='Method', marker='o')
    plt.title('Model Accuracy vs Number of Features for Different Feature Selection Methods')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.legend(title='Method')
    plt.savefig('output/plot/feature_selection_methods.png')

    ## Training the Model
    # Based on prev step rfe of size 15 gives the best accuracy
    best_feature_set = X.columns[RFE(estimator=model, n_features_to_select=15, step=1).fit(X, y_class).support_]
    
    data_train, data_test, _, _ = train_test_split(data, y_class, test_size=0.2, random_state=42, stratify=y_class)
    X_train = data_train[best_feature_set]
    y_train = data_train['Relative 2023Q2 Label']
    X_test = data_test[best_feature_set]
    y_test = data_test['Relative 2023Q2 Label']
    stocks_test = data_test['Stock'].reset_index(drop=True)

    # i) [Random Forest]
    forert_model = RandomForestClassifier(n_estimators=100, random_state=42)
    forert_model.fit(X_train, y_train)

    forest_preds = forert_model.predict(X_test)
    
    print("Random Forest Performance:")
    evaluate_classification(y_test, forest_preds)
    
    disp = ConfusionMatrixDisplay.from_estimator(forert_model, X_test, y_test, cmap=plt.cm.Blues)
    disp.ax_.set_title("Random Forest Confusion Matrix")
    disp.ax_.tick_params(axis='both', which='major', labelsize=5)
    plt.savefig('output/plot/forest_confusion.png')

    # ii) [SVM]
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)

    svm_preds = svm_model.predict(X_test)

    print("SVM Performance:")
    evaluate_classification(y_test, svm_preds)

    disp = ConfusionMatrixDisplay.from_estimator(svm_model, X_test, y_test, cmap=plt.cm.Blues)
    disp.ax_.set_title("SVM Confusion Matrix")
    disp.ax_.tick_params(axis='both', which='major', labelsize=5)
    plt.savefig('output/plot/svm_confusion.png')

    # Saving the predictions along with the stocks name
    pd.DataFrame({
    'Stock': stocks_test,
    'True Label': y_test.reset_index(drop=True),
    'Random Forest Prediction': forest_preds,
    'SVM Prediction': svm_preds
    }).to_csv('output/classification_results.csv')



if __name__ == "__main__":
    main()
