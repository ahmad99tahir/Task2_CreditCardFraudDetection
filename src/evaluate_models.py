#This file contains the functions to evaluate the models.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report,confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model using various metrics, display the confusion matrix,
    classification report, and prediction for a specific example.

    Parameters:
    - model (object): The trained classification model.
    - X_test (array-like or DataFrame): Test features.
    - y_test (array-like or Series): True labels for the test set.

    Returns:
    None
    """
        
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Print metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Generate Classification report
    cr = classification_report(y_test, y_pred)
    print("Classification Report")
    print(cr)

    # Display prediction for a specific example with index 99
    specific_example_index = 99
    if len(y_test) > specific_example_index:
        specific_example_features = X_test.iloc[specific_example_index] if isinstance(X_test, pd.DataFrame) else X_test[specific_example_index]
        specific_true_label = y_test.iloc[specific_example_index] if isinstance(y_test, pd.Series) else y_test[specific_example_index]
        specific_predicted_label = y_pred[specific_example_index]

        print(f"\nSpecific Example {specific_example_index + 1}:")
        print(f"Features: {specific_example_features}")
        print(f"True Label: {specific_true_label}")
        print(f"Predicted Label: {specific_predicted_label}")
    else:
        print(f"\nSpecific Example {specific_example_index + 1}: Index out of range")
