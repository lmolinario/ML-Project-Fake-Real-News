# Importing necessary libraries
from abc import ABC, abstractmethod

#Modules fot texx representation
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences


# Global variables
NUM_WORDS = 500  # Maximum number of words for tokenization
PAD_LENGTH = 100  # Length to pad/truncate sequences to



# Abstract class for data representation strategies
class DataRepresentationStrategy(ABC):
    @abstractmethod
    def prepare_data(self, train, test):
        pass

# Tokenizer-based data representation strategy
class TokenizerRepresentation(DataRepresentationStrategy):
    def prepare_data(self, train, test):
        tokenizer = Tokenizer(num_words=NUM_WORDS)
        tokenizer.fit_on_texts(train)
        train_sequences = tokenizer.texts_to_sequences(train)
        test_sequences = tokenizer.texts_to_sequences(test)
        padded_train = pad_sequences(train_sequences, maxlen=PAD_LENGTH, padding='post', truncating='post')
        padded_test = pad_sequences(test_sequences, maxlen=PAD_LENGTH, padding='post', truncating='post')
        return padded_train, padded_test


# TextVectorization-based data representation strategy
class TextVectorizationRepresentation(DataRepresentationStrategy):
    def prepare_data(self, train, test):
        vectorizer = tf.keras.layers.TextVectorization(max_tokens=NUM_WORDS, output_mode='int')
        vectorizer.adapt(train)
        train_sequences = vectorizer(train).numpy()
        test_sequences = vectorizer(test).numpy()
        padded_train = pad_sequences(train_sequences, maxlen=PAD_LENGTH, padding='post', truncating='post')
        padded_test = pad_sequences(test_sequences, maxlen=PAD_LENGTH, padding='post', truncating='post')
        return padded_train, padded_test


# TF-IDF-based data representation strategy
class TFIDFRepresentation(DataRepresentationStrategy):
    def prepare_data(self, train, test):
        train_list = train.tolist()
        test_list = test.tolist()
        vectorizer = TfidfVectorizer(max_features=NUM_WORDS)
        vectorizer.fit(train_list)
        train_tfidf = vectorizer.transform(train_list).toarray()
        test_tfidf = vectorizer.transform(test_list).toarray()
        return train_tfidf, test_tfidf