from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def j48(x_train, y_train, x_test, y_test):
    # Create decision tree classifier
    j48_classifier = DecisionTreeClassifier(criterion="entropy", random_state=43)

    # Model training
    j48_classifier.fit(x_train, y_train)

    # Make predictions
    y_pred = j48_classifier.predict(x_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Store metrics
    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report
    }

    return metrics