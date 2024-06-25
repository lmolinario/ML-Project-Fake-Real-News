# Importing necessary libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Initialize the lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def download_nltk_resources():
    """
    Download necessary NLTK resources for text preprocessing.
    """
    nltk.download('wordnet') #This line downloads the WordNet lexical database. WordNet is a large lexical database of English that groups words into sets of synonyms called synsets, providing short definitions and usage examples. It is commonly used for various NLP tasks, including synonym extraction and word sense disambiguation.
    nltk.download('averaged_perceptron_tagger') #This line downloads the averaged perceptron tagger model, which is used for part-of-speech tagging. POS tagging is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech based on its definition and context.
    nltk.download('punkt') #This line downloads the Punkt tokenizer models, which are used for sentence splitting. Punkt is a pre-trained tokenizer that can segment a text into sentences. It is essential for tasks that require sentence-level processing.
    nltk.download('stopwords') #This line downloads a list of common stopwords. Stopwords are words that are often filtered out in text processing because they occur frequently but don't carry significant meaning (e.g., "and", "the", "is").


def preprocessing_text(text): # took inspiration from  from https://www.kaggle.com/code/yossefmohammed/true-and-fake-news-lstm-accuracy-97-90?scriptVersionId=175166308&cellId=24
    """
    Preprocesses the input text by performing the following steps:
    1. Convert text to lowercase
    2. Replace non-alphanumeric characters with spaces
    3. Tokenize text into words
    4. Remove stopwords and words shorter than three characters
    5. Apply lemmatization and stemming

    Parameters:
    text (str): The input text to preprocess.

    Returns:
    list: A list of processed words.
    """
    # Download necessary NLTK data
    download_nltk_resources()

    # Save stopwords list
    stopwords_list = set(stopwords.words('english'))

    # Convert text to lowercase
    text_lower = text.casefold()

    # Replace non-alphanumeric characters with spaces
    text_cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", text_lower)

    # Tokenize the cleaned text into words
    words = word_tokenize(text_cleaned)

    # Filter out stopwords and words shorter than three characters
    filtered_words = [
        word for word in words
        if word not in stopwords_list and len(word) > 3
    ]

    # Apply lemmatization and stemming
    processed_words = [
        stemmer.stem(lemmatizer.lemmatize(word))
        for word in filtered_words
    ]

    return processed_words

