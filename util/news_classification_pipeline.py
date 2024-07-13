from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import numpy as np

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

    # Initialize metrics storage
    def initialize_metrics_storage(self, num_classifiers, num_representations):
        average_precision = np.zeros(shape=(num_classifiers, num_representations))
        average_recall = np.zeros(shape=(num_classifiers, num_representations))
        average_f1_score = np.zeros(shape=(num_classifiers, num_representations))
        average_confusion_matrix = np.zeros(shape=(num_classifiers, num_representations, 2, 2))
        return average_precision, average_recall, average_f1_score, average_confusion_matrix

    def train_and_evaluate(self):
        splitter = self.perform_kfold_split(self.x, self.y)
        average_precision, average_recall, average_f1_score, average_confusion_matrix = self.initialize_metrics_storage(
            1, 1)
        for i, (train_index, test_index) in enumerate(splitter):
            print(f"******************************* SPLIT {i+1} *******************************")
            x_train, x_test = self.data_representation.prepare_data(train=self.x[train_index], test=self.x[test_index])
            y_train, y_test = self.y[train_index], self.y[test_index]

            self.classifier.train(x_train, y_train)
            y_pred = self.classifier.predict(x_test)


            #Evaluation metrics
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, output_dict=False))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            average_confusion_matrix[0][0] += confusion_matrix(y_test, y_pred)
            average_precision[0][0] += report['macro avg']['precision']
            average_recall[0][0] += report['macro avg']['recall']
            average_f1_score[0][0] += report['macro avg']['f1-score']

        average_precision /= N_SPLITS
        average_recall /= N_SPLITS
        average_f1_score /= N_SPLITS
        average_confusion_matrix /= N_SPLITS

        return average_precision, average_recall, average_f1_score, average_confusion_matrix
    
    def determine_best_classifier(classifier_names, data_representation_names, average_f1_score):
        best_classifiers = {}
        for r_idx, data_rep_name in enumerate(data_representation_names):
            best_classifier = None
            highest_avg_f1 = 0
            for c_idx, classifier_name in enumerate(classifier_names):
                avg_f1 = average_f1_score[r_idx][c_idx]
                if avg_f1 > highest_avg_f1:
                    highest_avg_f1 = avg_f1
                    best_classifier = classifier_name
            best_classifiers[data_rep_name] = best_classifier
        return best_classifiers