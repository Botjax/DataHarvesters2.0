from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def random_forest(x_train, y_train, x_test, y_test):
    # Create random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=43)

    # train model
    rf_classifier.fit(x_train, y_train)

    # make predictions on test set
    y_pred = rf_classifier.predict(x_test)

    # get metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # put metrics into dictionary
    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report
    }

    return metrics