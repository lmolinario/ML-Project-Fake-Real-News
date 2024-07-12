from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# Global variables
N_SPLITS = 3  # Number of splits for cross-validation

class NewsClassificationPipeline():
    """
    This class implements a pipeline that takes as inputs a classifier and a preprocessed set of news. Given the set of
    news this class allows to split the data into test set and evaluation set.
    """
    def __init__(self, classifier, data_representation, dataset):
        self.classifier = classifier
        self.data_representation = data_representation
        self.dataset = dataset
        self.x, self.y = self.get_features_and_labels(self.dataset)

    @staticmethod
    def get_features_and_labels(dataset):
        x = dataset.filtered_string
        y = dataset.label
        return x, y

    def perform_kfold_split(self, x, y):
        return KFold(n_splits=N_SPLITS, random_state=42, shuffle=True).split(x, y)
    def train_and_evaluate(self):
        splitter = self.perform_kfold_split(self.x, self.y)
        for i, (train_index, test_index) in enumerate(splitter):
            print(f"******************************* SPLIT {i+1} *******************************")
            x_train, x_test = self.data_representation.prepare_data(train=self.x[train_index], test=self.x[test_index])
            y_train, y_test = self.y[train_index], self.y[test_index]

            self.classifier.train(x_train, y_train)
            y_pred = self.classifier.predict(x_test)

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, output_dict=False))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
