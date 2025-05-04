import pandas as pd
from sklearn.model_selection import train_test_split
from random_forest import random_forest
from knn import knn
from svm import svm
from j48 import j48
from relief_fs import relief_logistic
from elm import elm

def load_data(file_path):
    data = pd.read_csv(file_path)
    x = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    return train_test_split(x, y, test_size=0.2, random_state=42)

def display_metrics(model, metrics, neighbors=0):
    if neighbors == 0: print(f"\n=== {model} Results ===")
    else: print(f"\n=== {model} Results with {neighbors} Neighbors ===")
    print("Accuracy:", metrics["accuracy"])
    print("Confusion Matrix:\n", metrics["confusion_matrix"])
    print("Classification Report:")
    for label, scores in metrics["classification_report"].items():
        if isinstance(scores, dict):
            print(f"{label}: Precision: {scores['precision']:.2f}, Recall: {scores['recall']:.2f}, F1: {scores['f1-score']:.2f}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("Resources/processed_data/processed_ckd_onehot.csv")
    rf_metrics = random_forest(X_train, y_train, X_test, y_test)
    j48_metrics = j48(X_train, y_train, X_test, y_test)
    display_metrics("Random Forest", rf_metrics)
    display_metrics("J-48", j48_metrics)
    for i in range(1,6):
        knn_metrics = knn(X_train, y_train, X_test, y_test, neighbors=i)
        display_metrics("KNN", knn_metrics, neighbors=i)
    svm_metrics = svm(X_train, y_train, X_test, y_test)
    display_metrics("SVM", svm_metrics)
    relief_metrics = relief_logistic(X_train, y_train, X_test, y_test, top_k=10)
    display_metrics("Relief Logistic Regression", relief_metrics)
    elm_metrics = elm(X_train, y_train, X_test, y_test, hidden_neurons=100)
    display_metrics("ELM", elm_metrics)