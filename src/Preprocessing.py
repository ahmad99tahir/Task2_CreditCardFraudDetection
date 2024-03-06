#This file contains code to load the data and perform preprocessing on it.
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load a CSV dataset from the specified file path.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded dataset.
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the dataset by scaling features, splitting into training and testing sets, 
    and applying oversampling, SMOTE, and undersampling techniques.

    Parameters:
    - data (pd.DataFrame): Input dataset.

    Returns:
    Tuple: Tuple containing training and testing sets for various preprocessing techniques.
    """
    # Assume 'Class' is the target variable
    X = data.drop(['Class','Time'], axis=1)
    y = data['Class']

    # Scale the data using Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the scaled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Apply oversampling and undersampling to the scaled data
    oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

    X_train_over, y_train_over = oversampler.fit_resample(X_train, y_train)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

    return (X_train, y_train), (X_test, y_test), (X_train_over, y_train_over), (X_train_smote, y_train_smote), (X_train_under, y_train_under)

