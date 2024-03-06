#This file contains functions to perform Logistic Regression with Gradient Descent from scratch
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc,precision_score,recall_score,f1_score,roc_auc_score
import matplotlib.pyplot as plt

def save_logreg_model(theta, filepath):
    """
    Save the learned parameters of the logistic regression model to a specified file.

    Parameters:
    - theta (numpy array): The learned parameters of the logistic regression model.
    - filepath (str): The file path where the model parameters will be saved.

    Returns:
    - None
    """
    np.save(filepath, theta)

def load_logreg_model(filepath):
    """
    Load the learned parameters of the logistic regression model from a specified file.

    Parameters:
    - filepath (str): The file path from which the model parameters will be loaded.

    Returns:
    - numpy array: The loaded parameters of the logistic regression model.
    """
    return np.load(filepath)

def sigmoid(z):
    """
    Compute the sigmoid function element-wise for the given input.

    Parameters:
    - z (numpy array): The input to the sigmoid function.

    Returns:
    - numpy array: The output of the sigmoid function applied to each element of z.
    """

    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate, num_iterations):
    """
    Train a logistic regression model using gradient descent.

    Parameters:
    - X (numpy array): The feature matrix.
    - y (numpy array): The target vector.
    - learning_rate (float): The learning rate for gradient descent.
    - num_iterations (int): The number of iterations for gradient descent.

    Returns:
    - numpy array: The learned parameters of the logistic regression model.
    """
    m, n = X.shape
    X = np.insert(X, 0, 1, axis=1)

    theta = np.zeros(n + 1)

    for iteration in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        theta -= learning_rate * gradient

    return theta

def predict_logreg(X, theta):
    """
    Make binary predictions using the logistic regression model.

    Parameters:
    - X (numpy array): The feature matrix for making predictions.
    - theta (numpy array): The learned parameters of the logistic regression model.

    Returns:
    - numpy array: Binary predictions (0 or 1) based on the logistic regression model.
    """
    X = np.insert(X, 0, 1, axis=1)
    probabilities = sigmoid(np.dot(X, theta))
    return (probabilities >= 0.5).astype(int)

def calculate_logreg_metrics(y_true, y_pred):
    """
    Calculate precision, recall, F1 score, and ROC AUC score for a logistic regression model.

    Parameters:
    - y_true (numpy array): The true labels.
    - y_pred (numpy array): The predicted labels.

    Returns:
    - tuple: Precision, recall, F1 score, and ROC AUC score.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return precision, recall, f1, roc_auc

def plot_logreg_roc_curve(y_true, y_scores):
    """
    Plot the Receiver Operating Characteristic (ROC) curve for a logistic regression model.

    Parameters:
    - y_true (numpy array): The true labels.
    - y_scores (numpy array): The predicted scores.

    Returns:
    - None
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def plot_logreg_precision_recall_curve(y_true, y_scores):
    """
    Plot the Precision-Recall curve for a logistic regression model.

    Parameters:
    - y_true (numpy array): The true labels.
    - y_scores (numpy array): The predicted scores.

    Returns:
    - None
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

def plot_logreg_confusion_matrix(y_true, y_pred):
    """
    Plot the confusion matrix for a logistic regression model.

    Parameters:
    - y_true (numpy array): The true labels.
    - y_pred (numpy array): The predicted labels.

    Returns:
    - None
    """
    cm = confusion_matrix(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def generate_classification_report(y_true, y_pred):
    """
    Generate and print the classification report for a logistic regression model.

    Parameters:
    - y_true (numpy array): The true labels.
    - y_pred (numpy array): The predicted labels.

    Returns:
    - None
    """
    report = classification_report(y_true, y_pred)
    print("Classification Report:\n", report)

def grid_search_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Perform a grid search for the best learning rate for logistic regression using cross-validation.

    Parameters:
    - X_train (numpy array or pandas DataFrame): Training features.
    - y_train (numpy array or pandas Series): Training labels.
    - X_test (numpy array or pandas DataFrame): Testing features.
    - y_test (numpy array or pandas Series): Testing labels.

    Returns:
    - float: The best learning rate found during the grid search.
    """
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1]
    best_model = None
    best_learning_rate = None
    best_f1_score = 0.0

    # Convert X_train and y_train to pandas DataFrame or Series
    X_train_df = pd.DataFrame(X_train)
    y_train_series = pd.Series(y_train)

    # Use StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for learning_rate in learning_rates:
            # Train logistic regression with the current learning rate
            f1_scores = []
            for train_index, val_index in cv.split(X_train_df, y_train_series):
                X_train_fold, X_val_fold = X_train_df.iloc[train_index], X_train_df.iloc[val_index]
                y_train_fold, y_val_fold = y_train_series.iloc[train_index], y_train_series.iloc[val_index]

                theta = logistic_regression(X_train_fold.values, y_train_fold.values, learning_rate, num_iterations=1000)
                predictions = predict_logreg(X_val_fold.values, theta)
                f1_scores.append(f1_score(y_val_fold.values, predictions))

            avg_f1_score = np.mean(f1_scores)

            # Update best model and learning rate if the current learning rate gives a higher F1 score
            if avg_f1_score > best_f1_score:
                best_f1_score = avg_f1_score
                best_learning_rate = learning_rate
                best_model = logistic_regression(X_train_df.values, y_train_series.values, best_learning_rate, num_iterations=1000)

    # Save the best model after the loop
    save_logreg_model(best_model, 'models/logistic_regression_gradient_descent_practice.npy')

    # Evaluate the best model and print metrics
    y_pred = predict_logreg(X_test, best_model)
    precision, recall, f1, roc_auc = calculate_logreg_metrics(y_test, y_pred)
    print(f"Best Model Metrics:")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, ROC AUC: {roc_auc:.2f}")

    # Plot ROC curve, precision-recall curve, and confusion matrix for the best model
    plot_logreg_roc_curve(y_test, y_pred)
    plot_logreg_precision_recall_curve(y_test, y_pred)
    plot_logreg_confusion_matrix(y_test, y_pred)

    # Generate and print the classification report for the best model
    generate_classification_report(y_test, y_pred)

    return best_learning_rate