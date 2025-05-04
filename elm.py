import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer

def tanh(x):
    return np.tanh(x)

def elm(X_train, y_train, X_test, y_test, hidden_neurons=100):
    input_dim = X_train.shape[1]

    # Randomly initialize input weights and biases
    W = np.random.randn(input_dim, hidden_neurons)
    b = np.random.randn(hidden_neurons)

    # Hidden layer output matrix (training)
    H = tanh(np.dot(X_train, W) + b)

    # Compute output weights (Moore–Penrose pseudoinverse)
    H_pinv = np.linalg.pinv(H)
    lb = LabelBinarizer()
    y_train_bin = lb.fit_transform(y_train)
    beta = np.dot(H_pinv, y_train_bin)

    # Hidden layer output matrix (testing)
    H_test = tanh(np.dot(X_test, W) + b)  # Corrected to use X_test here
    y_pred_probs = np.dot(H_test, beta)
    y_pred = lb.inverse_transform(y_pred_probs)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }
    return metrics
