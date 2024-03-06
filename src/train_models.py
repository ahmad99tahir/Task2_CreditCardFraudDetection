#This file contains code to save, load, and train the different machine learning models.
import os
import joblib 
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score

MODEL_DIR = 'models'  # Folder to save the models

def create_model_directory():
    """
    Create the 'models' directory if it doesn't exist.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def save_model(model, model_name, save=True):
    """
    Save the trained model to the 'models' directory if save parameter is True.

    Parameters:
    - model: Trained machine learning model.
    - model_name (str): Name of the model.
    - save (bool): Flag to decide whether to save the model. Default is True.
    """
    if save:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved at: {model_path}")

def load_model(model_name):
    """
    Load a trained model from the 'models' directory.

    Parameters:
    - model_name (str): Name of the model.

    Returns:
    Trained machine learning model or None if the model is not found.
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        print(f"{model_name} model not found.")
        return None

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model using GridSearchCV for hyperparameter tuning.

    Parameters:
    - X_train: Features of the training set.
    - y_train: Target variable of the training set.

    Returns:
    Trained logistic regression model.
    """
    model = LogisticRegression()

    # Define the parameter grid for GridSearchCV
    param_grid = {'C': [0.0001,0.001, 0.01, 0.1, 1, 10, 100,1000]}

    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grid, scoring=make_scorer(f1_score), cv=5)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    # Return the best model
    return grid_search.best_estimator_

def train_random_forest(X_train, y_train):
    """
    Train a random forest model using GridSearchCV for hyperparameter tuning.

    Parameters:
    - X_train: Features of the training set.
    - y_train: Target variable of the training set.

    Returns:
    Trained random forest model.
    """
    model = RandomForestClassifier()

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grid, scoring=make_scorer(f1_score), cv=5)
    grid_search.fit(X_train, y_train)

    # Return the best model
    return grid_search.best_estimator_

def train_gradient_boosting(X_train, y_train):
    """
    Train a gradient boosting model using GridSearchCV for hyperparameter tuning.

    Parameters:
    - X_train: Features of the training set.
    - y_train: Target variable of the training set.

    Returns:
    Trained gradient boosting model.
    """
    model = GradientBoostingClassifier()

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grid, scoring=make_scorer(f1_score), cv=5)
    grid_search.fit(X_train, y_train)

    # Return the best model
    return grid_search.best_estimator_

def train_decision_tree(X_train, y_train):
    """
    Train a decision tree model using GridSearchCV for hyperparameter tuning.

    Parameters:
    - X_train: Features of the training set.
    - y_train: Target variable of the training set.

    Returns:
    Trained decision tree model.
    """
    model = DecisionTreeClassifier()

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grid, scoring=make_scorer(f1_score), cv=5)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    # Return the best model
    return grid_search.best_estimator_

