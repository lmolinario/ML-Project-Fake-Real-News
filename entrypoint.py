# Import necessary modules
from util import import_dataset
from util.create_db_news import create_db_news as crdb
import platform
import getpass

from util.nlp import word_tokenize, preprocessing_text #NLP modules
from util.import_dataset import  import_dataset
import util.db_analysis as dba

# Libraries to manage datasets
import os
import pandas as pd

def main():
    # Try to read the datasets using UNIX-like path
    system = platform.system()
    if system == 'Windows':
        current_dir = os.getcwd()
        true_file_path = os.path.join(current_dir, 'train', 'True.csv')
        fake_file_path = os.path.join(current_dir, 'train', 'Fake.csv')
        if not (os.path.isfile(true_file_path) and os.path.isfile(fake_file_path)):
            import_dataset()
    else:
        true_file_path = "./train/True.csv"
        fake_file_path = "./train/Fake.csv"
        if not (os.path.isfile(true_file_path) and os.path.isfile(fake_file_path)):
            import_dataset()

    ds_true = pd.read_csv(true_file_path)
    ds_fake = pd.read_csv(fake_file_path)


    # Add a label column to the datasets
    ds_true['label'] = 0  # 0 = true news
    ds_fake['label'] = 1  # 1 = fake news

    print("True News Dataset Preview:")
    print(ds_true.head())

    print("Fake News Dataset Preview:")
    print(ds_fake.head())

    # Flag to indicate whether to create the news database
    create_db_news = False

    # Try to load the preprocessed news dataset
    try:
        try:
            ds_news = pd.read_csv("News.csv")
            print("Preprocessed dataset loaded successfully!")
        except:
            current_dir = os.getcwd()
            news_file_path = os.path.join(current_dir, 'News.csv')
            ds_news = pd.read_csv(news_file_path)
            print("Preprocessed dataset loaded successfully!")
    except:
        print("Preprocessed dataset not found. Creating it...")
        create_db_news = True

    # If preprocessed dataset is not found, create it
    if create_db_news:
        ds_news = crdb(ds_true, ds_fake)

    print("News Dataset Preview:")
    print(ds_news.head())

    #Statistical analysis
    total_words = dba.get_unique_word_count(ds_news.filtered) #Count total unique words across all documents
    print("Total unique words", total_words)

    maxlen, imax, minlen, imin = dba.get_max_min_word_count(ds_news.filtered) #Document with the highest and lowest number of words
    print(f"Document with max words (length {maxlen}): {imax}")
    print(f"Document with min words (length {minlen}): {imin}")

    maxdim, imaxu, mindim, iminu = dba.get_max_min_unique_word_count(ds_news.filtered_unique) #Document with highest and lowest number of unique words
    print(f"Document with max unique words (length {maxdim}): {imaxu}")
    print(f"Document with min unique words (length {mindim}): {iminu}")

# Execute the main function
if __name__ == "__main__":
    main()
