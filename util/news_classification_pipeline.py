
class NewsClassificationPipeline():
    """
    This class implements a pipeline that takes as inputs a classifier and a preprocessed set of news. Given the set of
    news this class allows to split the data into test set and evaluation set.
    """
    def __init__(self, classifier, data):
        self.classifier = classifier
        self.data = data