# Import necessary modules
from util import import_db
from util.create_db_news import create_db_news as crdb

# Libraries to manage datasets
import os
import pandas as pd

def main():
    # Try to read the datasets using UNIX-like path
    try:
        ds_true = pd.read_csv("./train/True.csv")
        ds_fake = pd.read_csv("./train/Fake.csv")
    except:  # If failed, try Windows path
        current_dir = os.getcwd()
        true_file_path = os.path.join(current_dir, 'train', 'True.csv')
        fake_file_path = os.path.join(current_dir, 'train', 'Fake.csv')

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

# Execute the main function
if __name__ == "__main__":
    main()
