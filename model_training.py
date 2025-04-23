import pandas as pd
from sklearn.model_selection import train_test_split
from random_forest import random_forest

def load_data(file_path):
    data = pd.read_csv(file_path)
    x = data.drop('target', axis=1)
    y = data['target']
    return train_test_split(x, y, test_size=0.2, random_state=42)

def display_metrics(model, metrics):
    print(f"\n=== {model} Results ===")
    print("Accuracy:", metrics["accuracy"])
    print("Confusion Matrix:\n", metrics["confusion_matrix"])
    print("Classification Report:")
    for label, scores in metrics["report"].items():
        if isinstance(scores, dict):
            print(f"{label}: Precision: {scores['precision']:.2f}, Recall: {scores['recall']:.2f}, F1: {scores['f1-score']:.2f}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("changeLater.csv")
    rf_metrics = random_forest(X_train, y_train, X_test, y_test)
    j48_metrics = j48(X_train, y_train, X_test, y_test)
    display_metrics("Random Forest", rf_metrics)
    display_metrics("J-48", j48_metrics)