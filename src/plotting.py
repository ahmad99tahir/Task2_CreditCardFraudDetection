#this file contains the funcitons to plot the data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

def plot_roc_curve(model, X_test, y_test):
    """
    Plot the Receiver Operating Characteristic (ROC) curve for a binary classification model.

    Parameters:
    - model: Trained binary classification model.
    - X_test (array-like or pd.DataFrame): Test features.
    - y_test (array-like or pd.Series): True labels for the test set.

    Returns:
    None
    """
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc = 'lower right')
    plt.show()

def plot_precision_recall_curve(model, X_test, y_test):
    """
    Plot the Precision-Recall curve for a binary classification model.

    Parameters:
    - model: Trained binary classification model.
    - X_test (array-like or pd.DataFrame): Test features.
    - y_test (array-like or pd.Series): True labels for the test set.

    Returns:
    None
    """
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Generate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    # Plot precision-recall curve
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

'''def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()'''

