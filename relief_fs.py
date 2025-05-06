import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def relief(X, y, n_iterations=None):
    y = np.asarray(y)
    n_samples, n_features = X.shape
    scores = np.zeros(n_features)

    if n_iterations is None:
        n_iterations = n_samples

    for _ in range(n_iterations):
        idx = np.random.randint(0, n_samples)
        sample, label = X[idx], y[idx]

        distances = euclidean_distances([sample], X)[0]
        distances[idx] = np.inf

        hit_idx = np.argmin([d if y[i] == label else np.inf for i, d in enumerate(distances)])
        miss_idx = np.argmin([d if y[i] != label else np.inf for i, d in enumerate(distances)])

        hit, miss = X[hit_idx], X[miss_idx]

        for j in range(n_features):
            scores[j] -= (sample[j] - hit[j])**2 / n_iterations
            scores[j] += (sample[j] - miss[j])**2 / n_iterations

    return scores


def relief_logistic(X_train, y_train, X_test, y_test, top_k=10):
    # Standardize both sets using training set's scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Run Relief on training data only
    relief_scores = relief(X_train_scaled, y_train)
    top_indices = np.argsort(relief_scores)[-top_k:]

    # Select top-k features from both sets
    X_train_selected = X_train_scaled[:, top_indices]
    X_test_selected = X_test_scaled[:, top_indices]

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    feature_importances = X_train.columns[top_indices]
    feature_scores = relief_scores[top_indices]

    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'feature_importances': pd.DataFrame({
            "Feature": feature_importances,
            "Score": feature_scores
        })
    }

    return metrics
