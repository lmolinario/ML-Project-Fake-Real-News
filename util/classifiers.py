# Modules for Classifier
from abc import ABC, abstractmethod
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

# Abstract class for classifier strategies
class ClassifierStrategy(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

# Naive Bayes Classifier strategy
class NaiveBayesClassifierStrategy(ClassifierStrategy):
    def __init__(self):
        self.model = MultinomialNB()

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

class MultiLayerPerceptronNet(ClassifierStrategy):
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(512,10), max_iter=1500)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)