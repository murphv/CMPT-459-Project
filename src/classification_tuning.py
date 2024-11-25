import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
    mean_squared_error, mean_absolute_error, r2_score
)
import seaborn as sns


def evaluate_classification(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    report = f'Classification Results for {model_name}:\n'
    report += f"Accuracy: {accuracy:.4f}\n"
    report += f"Precision: {precision:.4f}\n"
    report += f"Recall: {recall:.4f}\n"
    report += f"F1-Score: {f1:.4f}\n"
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    report += "Confusion Matrix:\n"
    report += cm_df.to_string() + "\n\n"
    report += "\n\n Classification Report: \n"
    report += classification_report(y_true, y_pred)

     # Classification Report
    cls_report = classification_report(y_true, y_pred)
    report += "Classification Report:\n"
    report += cls_report

    with open(f'output/{model_name}_classification.txt', 'w', encoding='utf-8') as f:
        f.write(report)


def evaluate_regression(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    report = f'Regression Results for {model_name}:\n'
    report += f"Mean Squared Error (MSE): {mse:.4f}\n"
    report += f"Mean Absolute Error (MAE): {mae:.4f}\n"
    report += f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
    report += f"R^2 Score: {r2:.4f}\n"

    with open(f'output/{model_name}_regression.txt', 'w', encoding='utf-8') as f:
        f.write(report)


def main():
    data = pd.read_csv('data/normalized_encoded_data.csv')

    X = data.copy()
    X = X.drop(columns=['Relative 2023Q2', 'Relative 2023Q2 Label', 'Stock'])
    y_class = data['Relative 2023Q2 Label']
    y_reg = data['Relative 2023Q2']

    # Getting the features selected by feature selection methods
    cls_features = pd.read_csv('data/best_features_classification.csv')['Feature'].unique()
    reg_features = pd.read_csv('data/best_features_regression.csv')['Feature'].unique()


    ## ===== Classification =====
    data_train, data_test, _, _ = train_test_split(data, y_class, test_size=0.2, random_state=42, stratify=y_class)
    X_train = data_train[cls_features]
    y_train = data_train['Relative 2023Q2 Label']
    X_test = data_test[cls_features]
    y_test = data_test['Relative 2023Q2 Label']
    stocks_test = data_test['Stock'].reset_index(drop=True)

    # i) [Random Forest]
    forest_grid_param = {
        'n_estimators': [100, 200, 500, 100],
        'max_depth': [None, 10, 25 ,100],
        'min_samples_split': [2, 5, 10, 25]
    }
    forert_model_cls = RandomForestClassifier(random_state=42)
    forest_grid_search_cls = GridSearchCV(
        estimator=forert_model_cls,
        param_grid=forest_grid_param,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    forest_grid_search_cls.fit(X_train, y_train)
    best_forest = forest_grid_search_cls.best_estimator_
    forest_preds_cls = best_forest.predict(X_test)
    evaluate_classification(y_test, forest_preds_cls, 'forest')

    plt.figure(figsize=(16, 12))
    sns.lineplot(
        data=forest_grid_search_cls.cv_results_,
        x='param_n_estimators',
        y='mean_test_score',
        hue='param_max_depth',
        style='param_min_samples_split',
        markers=True,
        dashes=False
    )
    plt.title('Random Forest Classifier Hyperparameter Tuning')
    plt.xlabel('Number of Trees (n_estimators)')
    plt.ylabel('Mean Test Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('output/plot/rf_classifier_tuning.png', bbox_inches='tight')
    plt.close()
    
    disp = ConfusionMatrixDisplay.from_estimator(best_forest, X_test, y_test, cmap=plt.cm.Blues)
    disp.ax_.set_title("Random Forest Confusion Matrix")
    disp.ax_.tick_params(axis='both', which='major', labelsize=5)
    plt.savefig('output/plot/forest_confusion.png')

    # ii) [SVM]
    param_grid_svm = {
        'C': [1, 5, 10, 20],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear', 'sigmoid', 'poly']
    }
    svc = SVC(probability=True, random_state=42)
    grid_search_svc = GridSearchCV(
        estimator=svc,
        param_grid=param_grid_svm,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search_svc.fit(X_train, y_train)
    best_svc = grid_search_svc.best_estimator_
    svc_preds = best_svc.predict(X_test)
    evaluate_classification(y_test, svc_preds, 'svc')

    plt.figure(figsize=(16, 12))
    svc_results = pd.DataFrame(grid_search_svc.cv_results_)
    pivot_table = svc_results.pivot_table(
        values='mean_test_score',
        index='param_C',
        columns='param_kernel'
    )
    sns.heatmap(pivot_table, annot=True, fmt=".4f")
    plt.title('SVM Classifier Hyperparameter Tuning')
    plt.xlabel('Kernel')
    plt.ylabel('C')
    plt.savefig('output/plot/svc_classifier_tuning.png', bbox_inches='tight')
    plt.close()
    

    # iii) [Baseline Classifier - Random Labels]
    np.random.seed(42)
    random_preds_cls = np.random.choice(y_train.unique(), size=len(y_test))
    evaluate_classification(y_test, random_preds_cls, 'baseline_random')

    disp = ConfusionMatrixDisplay.from_estimator(best_svc, X_test, y_test, cmap=plt.cm.Blues)
    disp.ax_.set_title("SVM Confusion Matrix")
    disp.ax_.tick_params(axis='both', which='major', labelsize=5)
    plt.savefig('output/plot/svm_confusion.png')

    # Saving the predictions along with the stocks name
    pd.DataFrame({
    'Stock': stocks_test,
    'True Label': y_test.reset_index(drop=True),
    'Forest Prediction': forest_preds_cls,
    'SVM Prediction': svc_preds
    }).sort_values(by='Stock').to_csv('output/classification_results.csv')


    ## ====== Regression ======
    data_train, data_test, _, _ = train_test_split(data, y_reg, test_size=0.2, random_state=42)
    X_train = data_train[reg_features]
    y_train = data_train['Relative 2023Q2']
    X_test = data_test[reg_features]
    y_test = data_test['Relative 2023Q2']
    stocks_test = data_test['Stock'].reset_index(drop=True)

    # i) [Random Forest]
    forest_reg = RandomForestRegressor(random_state=42)
    forest_grid_search_reg = GridSearchCV(
        estimator=forest_reg,
        param_grid=forest_grid_param,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    forest_grid_search_reg.fit(X_train, y_train)
    best_forest = forest_grid_search_reg.best_estimator_
    forest_preds = best_forest.predict(X_test)
    evaluate_regression(y_test, forest_preds, 'forest')

    rf_reg_results = pd.DataFrame(forest_grid_search_reg.cv_results_)
    plt.figure(figsize=(16,12))
    sns.lineplot(
        data=rf_reg_results,
        x='param_n_estimators',
        y=-rf_reg_results['mean_test_score'],  # Negative MSE to MSE
        hue='param_max_depth',
        style='param_min_samples_split',
        markers=True,
        dashes=False
    )
    plt.title('Random Forest Regressor Hyperparameter Tuning')
    plt.xlabel('Number of Trees (n_estimators)')
    plt.ylabel('Mean Test MSE')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('output/plot/rf_regressor_tuning.png', bbox_inches='tight')
    plt.close()

    # ii) [SVM]
    param_grid_svr = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'epsilon': [0.1, 0.2, 0.5],
        'kernel': ['rbf', 'linear', 'sigmoid', 'poly']
    }
    svr = SVR()
    grid_search_svr = GridSearchCV(
        estimator=svr,
        param_grid=param_grid_svr,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search_svr.fit(X_train, y_train)
    best_svr = grid_search_svr.best_estimator_
    svr_preds = best_svr.predict(X_test)
    evaluate_regression(y_test, svr_preds, 'svr')
    
    # Plotting Grid Search Results for SVR Regressor
    svr_results = pd.DataFrame(grid_search_svr.cv_results_)
    pivot_table = svr_results.pivot_table(
        values='mean_test_score',
        index='param_C',
        columns='param_kernel'
    )
    plt.figure(figsize=(16,12))
    sns.heatmap(-pivot_table, annot=True, fmt=".4f")  # Negative MSE to MSE
    plt.title('SVR Regressor Hyperparameter Tuning')
    plt.xlabel('Kernel')
    plt.ylabel('C')
    plt.savefig('output/plot/svr_regressor_tuning.png', bbox_inches='tight')
    plt.close()
    

    # iii) [Baseline Regressor - Predict Zero]
    np.random.seed(42)
    random_preds_reg = np.random.uniform(-200, 200, size=len(y_test))
    evaluate_regression(y_test, random_preds_reg, 'baseline_random')

    # Saving the predictions along with the stocks name
    pd.DataFrame({
    'Stock': stocks_test,
    'True Value': y_test.reset_index(drop=True),
    'Forest Prediction': forest_preds,
    'SVM Prediction': svr_preds
    }).sort_values(by='True Value', ascending=False).to_csv('output/regression_results.csv')


if __name__ == "__main__":
    main()