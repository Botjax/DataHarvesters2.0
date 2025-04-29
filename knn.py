from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

def knn(x_train, y_train, x_test, y_test, neighbors=3):
    # Create and train KNN model
    knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(neighbors)) # default neighbors = 3
    knn_model.fit(x_train, y_train)

    # predict and evaluate
    y_pred = knn_model.predict(x_test)
   
    # get metrics
    accuracy = knn_model.score(x_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # put metrics into dictionary
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }
    return metrics