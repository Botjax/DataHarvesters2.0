# multiSURF_fs.py

import pandas as pd
from skrebate import MultiSURF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def multiSURF(X_train, y_train, X_test, y_test, top_k=10):
    # Fit MultiSURF
    fs = MultiSURF()
    fs.fit(X_train.values, y_train.values)

    # Get top K feature indices
    top_k_indices = fs.feature_importances_.argsort()[::-1][:top_k]
    selected_columns = X_train.columns[top_k_indices]

    # Select top K features
    X_train_selected = X_train[selected_columns]
    X_test_selected = X_test[selected_columns]

    # Train logistic regression on selected features
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("Resources/processed_data/processed_ckd_onehot.csv")

    # Adjust the column name below to match your actual label column
    target_col = "Diagnosis"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run MultiSURF + Logistic Regression
    metrics = multiSURF(X_train, y_train, X_test, y_test, top_k=10)

    # Print the results
    print("\n=== MultiSURF Logistic Regression Metrics===")
    print("Accuracy:", metrics["accuracy"])
    print("Confusion Matrix:\n", metrics["confusion_matrix"])
    print("Classification Report:")
    for label, scores in metrics["classification_report"].items():
        if isinstance(scores, dict):
            print(f"{label}: Precision: {scores['precision']:.2f}, Recall: {scores['recall']:.2f}, F1: {scores['f1-score']:.2f}")
