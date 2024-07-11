# Importing necessary libraries
from abc import ABC, abstractmethod




# Abstract class for data representation strategies
class DataRepresentationStrategy(ABC):
    @abstractmethod
    def prepare_data(self, train, test):
        pass





# Abstract class for classifier strategies
class ClassifierStrategy(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

