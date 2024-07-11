# Import necessary modules
import util.logo
#from util import import_dataset
from util.create_db_news import create_db_news as crdb
from util.import_dataset import import_dataset
import util.db_analysis as dba
from util.NewsPlot import plot_news_data

# Libraries to manage datasets
import os
import pandas as pd
import platform


def main():
    # Determine the operating system
    system = platform.system()

    # Set file paths based on the operating system
    if system == 'Windows':
        current_dir = os.getcwd()
        true_file_path = os.path.join(current_dir, 'train', 'True.csv')
        fake_file_path = os.path.join(current_dir, 'train', 'Fake.csv')
    else:
        true_file_path = "./train/True.csv"
        fake_file_path = "./train/Fake.csv"

    # Import datasets if they do not exist
    if not (os.path.isfile(true_file_path) and os.path.isfile(fake_file_path)):
        import_dataset()

    # Read the datasets into pandas DataFrames
    ds_true = pd.read_csv(true_file_path)
    ds_fake = pd.read_csv(fake_file_path)

    # Add a label column to each dataset
    ds_true['label'] = 0  # 0 = true news
    ds_fake['label'] = 1  # 1 = fake news

    # Display previews of the datasets
    print(f"True News Dataset Preview:\n{ds_true.head()}")
    print(f"\nFake News Dataset Preview:\n{ds_fake.head()}")

    # Flag to indicate whether to create the news database
    create_db_news = False

    # Attempt to load the preprocessed news dataset
    try:
        ds_news = pd.read_csv("News.csv")
        print("\nPreprocessed dataset loaded successfully!")
    except:
        try:
            current_dir = os.getcwd()
            news_file_path = os.path.join(current_dir, 'News.csv')
            ds_news = pd.read_csv(news_file_path)
            print("\nPreprocessed dataset loaded successfully!")
        except:
            print("Preprocessed dataset not found. Creating it...")
            create_db_news = True

    # Create the news dataset if it was not found
    if create_db_news:
        ds_news = crdb(ds_true, ds_fake)

    # Display a preview of the combined news dataset
    print(f"\nNews Dataset Preview:\n{ds_news.head()}")

    # Perform statistical analysis
    total_words = dba.get_unique_word_count(ds_news.filtered)  # Count total unique words across all documents
    print(f"\nTotal unique words: {total_words}")

    maxlen, imax, minlen, imin = dba.get_max_min_word_count(
        ds_news.filtered)  # Document with the highest and lowest number of words
    print(f"Document with max words (length {maxlen}): {imax}")
    print(f"Document with min words (length {minlen}): {imin}")

    maxdim, imaxu, mindim, iminu = dba.get_max_min_unique_word_count(
        ds_news.filtered_unique)  # Document with highest and lowest number of unique words
    print(f"Document with max unique words (length {maxdim}): {imaxu}")
    print(f"Document with min unique words (length {mindim}): {iminu}")

    # Plot the dataset information
    plot_news_data(ds_true, ds_fake, ds_news)


# Execute the main function
if __name__ == "__main__":
    main()
