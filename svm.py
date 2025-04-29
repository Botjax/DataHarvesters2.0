from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

def svm(x_train, y_train, x_test, y_test):
    # create and train SVM model
    svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale'))
    svm_model.fit(x_train, y_train)

    # predict and evaluate
    y_pred = svm_model.predict(x_test)
    
    # get metrics
    accuracy = svm_model.score(x_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # put metrics into dictionary
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }
    return metrics
