# Libraries to manage datasets
import pandas as pd
import os

# Libraries to process natural language
from util.nlp import preprocessing_text


def create_db_news(ds_true, ds_fake):
    """
    Merges true and fake news datasets, preprocesses text data, and saves the result to a CSV file.

    Args:
        ds_true (pd.DataFrame): DataFrame containing true news data.
        ds_fake (pd.DataFrame): DataFrame containing fake news data.

    Returns:
        pd.DataFrame: The merged and preprocessed news dataset.
    """
    # Merge the true and fake news datasets
    ds_news = pd.concat([ds_true, ds_fake]).reset_index(drop=True)

    # Remove the 'date' column as it is not useful
    ds_news.drop(columns=['date'], inplace=True)

    # Create a new column by merging 'title' and 'text' columns
    ds_news['title_text'] = ds_news['title'] + ' ' + ds_news['text']

    # Apply text preprocessing to the merged text
    ds_news['filtered'] = ds_news['title_text'].apply(preprocessing_text)

    # Convert the list of words into a single string
    ds_news['filtered_string'] = ds_news['filtered'].apply(lambda x: " ".join(x))

    # Add a column with a list of unique words for each row
    ds_news['filtered_unique'] = ds_news['filtered'].apply(lambda x: list(set(x)))

    # Drop columns 'filtered' and 'filtered_unique' before saving as CSV
    ds_news.drop(columns=['filtered', 'filtered_unique'], inplace=True)

    # Save the preprocessed dataset to a CSV file
    try:
        ds_news.to_csv("./News.csv", index=False)
    except Exception as e:
        current_dir = os.getcwd()
        news_file_path = os.path.join(current_dir, 'News.csv')
        ds_news = pd.read_csv(news_file_path)

    return ds_news

