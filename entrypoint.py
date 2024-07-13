# Import necessary modules
import util.logo
from util.create_db_news import create_db_news as crdb
from util.import_dataset import import_dataset
import util.db_analysis as dba
from util.news_plot import plot_news_data, plot_performance_evaluation
import util.text_representation as tr
import util.classifiers as cl
import util.news_classification_pipeline as cls_pipeline

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

    # Initialize the data representations and classifiers
    data_representations = [
        tr.TokenizerRepresentation(),
        tr.TextVectorizationRepresentation(),
        tr.TFIDFRepresentation()
        ]

    data_representation_names = ['Tokenizer', 'Vectorizer', 'TFIDF']

    classifiers = [
        cl.NaiveBayesClassifierStrategy(),
        cl.MultiLayerPerceptronNetStrategy(),
        cl.RandomForestClassifierStrategy()
    ]

    classifier_names = ['NaiveBayes', 'PerceptronNet', 'RandomForest']

    avg_precision=[]
    avg_recall=[]
    avg_f1_score=[]
    for data_rep, data_rep_name in zip(data_representations, data_representation_names):
        precisions = []
        recalls = []
        f1_scores = []
        for classifier, classifier_name in zip(classifiers, classifier_names):

            print(f"\n Test results for classifier: {classifier_name} with data representation: {data_rep_name}")

            # Initialize news classificaiton pipeline
            pipeline = cls_pipeline.NewsClassificationPipeline(classifier, data_rep, ds_news)

            # Train and evaluate the pipeline
            average_precision, average_recall, average_f1_score, average_confusion_matrix = pipeline.train_and_evaluate()

            # Store metrics
            precisions.append(average_precision[0, 0])
            recalls.append(average_recall[0, 0])
            f1_scores.append(average_f1_score[0, 0])

        # Store average metrics for this data representation
        avg_precision.append(precisions)
        avg_recall.append(recalls)
        avg_f1_score.append(f1_scores)

    # Print the data being plotted
    print("Plotting performance metrics for the following classifiers:")
    print(classifier_names)
    print("Across the following data representations:")
    print(data_representation_names)

    print("\nAverage Precision values:")
    for i, classifier_name in enumerate(classifier_names):
        precision_values = [avg_precision[j][i] for j in range(len(avg_precision))]
        print(f"{classifier_name}: {precision_values}")

    print("\nAverage Recall values:")
    for i, classifier_name in enumerate(classifier_names):
        recall_values = [avg_recall[j][i] for j in range(len(avg_recall))]
        print(f"{classifier_name}: {recall_values}")

    print("\nAverage F1-score values:")
    for i, classifier_name in enumerate(classifier_names):
        f1_values = [avg_f1_score[j][i] for j in range(len(avg_f1_score))]
        print(f"{classifier_name}: {f1_values}")

    plot_performance_evaluation(classifier_names,data_representation_names,avg_precision,avg_recall,avg_f1_score)


# Execute the main function
if __name__ == "__main__":
    main()
