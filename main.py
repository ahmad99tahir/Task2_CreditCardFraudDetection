from src.Preprocessing import load_data, preprocess_data
from src.train_models import train_logistic_regression, train_random_forest, train_gradient_boosting, train_decision_tree, save_model,load_model
from src.evaluate_models import evaluate_model
from src.plotting import plot_roc_curve, plot_precision_recall_curve
from src.logreg_gd import load_logreg_model, predict_logreg, calculate_logreg_metrics, plot_logreg_confusion_matrix, plot_logreg_precision_recall_curve, plot_logreg_roc_curve, generate_classification_report, grid_search_logistic_regression

def main():
    # Load and preprocess data
    file_path = 'Dataset/creditcard.csv' 
    data = load_data(file_path)
    (X_train, y_train), (X_test, y_test), (X_train_over, y_train_over), (X_train_smote, y_train_smote), (X_train_under, y_train_under) = preprocess_data(data)

    # Train and save your model
    """
    logistic_regression_under_model = train_logistic_regression(X_train_under, y_train_under)
    save_model(logistic_regression_under_model, 'logistic_regression_practice', save=True)
    """

    # Sample run to load and run a model
    logistic_regression_under_model = load_model('logistic_regression_under_f1')
    print("Logistic Regression Undersampled:")
    evaluate_model(logistic_regression_under_model, X_test, y_test)
    # Plotting
    plot_roc_curve(logistic_regression_under_model, X_test, y_test)
    plot_precision_recall_curve(logistic_regression_under_model, X_test, y_test)
    


    #Run the below line to train and evaluate a Logistic Regression Gradient Descent Model
    #grid_search_logistic_regression(X_train_under, y_train_under, X_test, y_test)

    # This is to load the model and perform Logistic Regression using gradient Descent
    """
    best_model = load_logreg_model('models/logistic_regression_gradient_descent.npy')
    #Make predictions on the test set using the best model
    y_pred = predict_logreg(X_test, best_model)
    # Evaluate the model and print metrics
    precision, recall, f1, roc_auc = calculate_logreg_metrics(y_test, y_pred)
    print("Final Model Metrics:")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, ROC AUC: {roc_auc:.2f}")
    # Plot ROC curve, precision-recall curve, and confusion matrix
    plot_logreg_roc_curve(y_test, y_pred)
    plot_logreg_precision_recall_curve(y_test, y_pred)
    plot_logreg_confusion_matrix(y_test, y_pred)
    # Generate and print the classification report
    generate_classification_report(y_test, y_pred)
    """
    

if __name__ == "__main__":
    main()
