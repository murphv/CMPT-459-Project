import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
    mean_squared_error, mean_absolute_error, r2_score
)


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
    forert_model_cls = RandomForestClassifier(n_estimators=100, random_state=42)
    forert_model_cls.fit(X_train, y_train)
    forest_preds_cls = forert_model_cls.predict(X_test)
    
    evaluate_classification(y_test, forest_preds_cls, 'forest')
    
    disp = ConfusionMatrixDisplay.from_estimator(forert_model_cls, X_test, y_test, cmap=plt.cm.Blues)
    disp.ax_.set_title("Random Forest Confusion Matrix")
    disp.ax_.tick_params(axis='both', which='major', labelsize=5)
    plt.savefig('output/plot/forest_confusion.png')

    # ii) [SVM]
    svc_model = SVC(kernel='rbf', probability=True, random_state=42)
    svc_model.fit(X_train, y_train)
    svc_preds = svc_model.predict(X_test)

    evaluate_classification(y_test, svc_preds, 'svm')

    # iii) [Baseline Classifier - Random Labels]
    np.random.seed(42)
    random_preds_cls = np.random.choice(y_train.unique(), size=len(y_test))
    evaluate_classification(y_test, random_preds_cls, 'baseline_random')

    disp = ConfusionMatrixDisplay.from_estimator(svc_model, X_test, y_test, cmap=plt.cm.Blues)
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
    forest_model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_model_reg.fit(X_train, y_train)
    forest_preds_reg = forest_model_reg.predict(X_test)
    evaluate_regression(y_test, forest_preds_reg, 'forest')

    # ii) [SVM]
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    svr_preds = svr_model.predict(X_test)
    evaluate_regression(y_test, svr_preds, 'svm')

    # iii) [Baseline Regressor - Predict Zero]
    np.random.seed(42)
    random_preds_reg = np.random.uniform(-50, 50, size=len(y_test))
    evaluate_regression(y_test, random_preds_reg, 'baseline_random')

    # Saving the predictions along with the stocks name
    pd.DataFrame({
    'Stock': stocks_test,
    'True Value': y_test.reset_index(drop=True),
    'Forest Prediction': forest_preds_reg,
    'SVM Prediction': svr_preds
    }).sort_values(by='True Value', ascending=False).to_csv('output/regression_results.csv')


if __name__ == "__main__":
    main()