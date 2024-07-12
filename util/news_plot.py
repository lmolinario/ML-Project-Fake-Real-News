import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

DB_PATH='plots/database_analysis'
PLOTS_PATH='plots'

def plot_news_data(ds_true, ds_fake, ds_news):
    # took inspiration from https://www.kaggle.com/code/nileshely/unmasking-truth-exploring-fake-and-real-news-data

    """
    This funciton takes as input three datasets, one containing just the true news, one the false news and one a mix of
    the two. Plots various information: Most common words in each class, number of subjects in each class and the
    proportion fo fake and true news in the set of mixed news.
    Args:
        ds_true: set of true news
        ds_fake: set of fake news
        ds_news: set composed of both true and fake news
    """

    # Ensure save_path exists
    os.makedirs(SAVE_PATH, exist_ok=True)  # Create folder if not exists

    # Create a figure and a grid of subplots with 1 row and 3 columns
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))
    fig.suptitle("Most common words", fontsize=36)

    # Word-cloud graph of the true news
    axes[0].set_title("True News")
    wc = WordCloud(max_words=300, width=1600, height=800).generate(
        " ".join(ds_news[ds_news.label == 0].filtered_string))
    axes[0].imshow(wc, interpolation='bilinear')

    # Word-cloud graph of the true news
    axes[1].set_title("Fake News")
    wc = WordCloud(max_words=300, width=1600, height=800).generate(
        " ".join(ds_news[ds_news.label == 1].filtered_string))
    axes[1].imshow(wc, interpolation='bilinear')

    # Display and save the plots
    plt.savefig(os.path.join(SAVE_PATH, 'most_common_words.png'))
    plt.show()
    plt.close(fig)

    # Create a figure and a grid of subplots with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle("News subjects", fontsize=36)

    # Plot for True.csv subjects
    axes[0].set_title("Subject Count True News dataset")
    sns.countplot(ax=axes[0], x="subject", data=ds_true)

    # Plot for Fake.csv subjects
    axes[1].set_title("Subject Count Fake News dataset")
    sns.countplot(ax=axes[1], x="subject", data=ds_fake)

    # Plot for the entire dataset subjects
    axes[2].set_title("Subject Count Total")
    sns.countplot(ax=axes[2], x="subject", data=ds_news)

    # Display and save the plots
    plt.savefig(os.path.join(SAVE_PATH, 'news_subjects.png'))

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    # Count-plot of label values
    plt.figure(figsize=(8, 8))
    plt.title("Label Count")
    sns.countplot(x="label", data=ds_news)

    # Display and save the plots
    plt.savefig(os.path.join(SAVE_PATH, 'label_count.png'))

    plt.show()
    plt.close(fig)

def plot_performance_evaluation(classifier_names,data_representation_names,avg_precision,avg_recall,avg_f1_score):
    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot about average precision
    plt.subplot(1, 3, 1)
    for i, classifier_name in enumerate(classifier_names):
        plt.plot(data_representation_names, [avg_precision[j][i] for j in range(len(avg_precision))], marker='o',
                 label=classifier_name)
    plt.title('Average Precision')
    plt.xlabel('Data Representation')
    plt.ylabel('Precision')
    plt.legend()

    # Plot average recall
    plt.subplot(1, 3, 2)
    for i, classifier_name in enumerate(classifier_names):
        plt.plot(data_representation_names, [avg_recall[j][i] for j in range(len(avg_recall))], marker='o',
                 label=classifier_name)
    plt.title('Average Recall')
    plt.xlabel('Data Representation')
    plt.ylabel('Recall')
    plt.legend()

    # Plot average F1-score
    plt.subplot(1, 3, 3)
    for i, classifier_name in enumerate(classifier_names):
        plt.plot(data_representation_names, [avg_f1_score[j][i] for j in range(len(avg_f1_score))], marker='o',
                 label=classifier_name)
    plt.title('Average F1-score')
    plt.xlabel('Data Representation')
    plt.ylabel('F1-score')
    plt.legend()

    filename='performance_metrics_plot.png'
    plot_dir = PERFORMANCE_PATH

    os.makedirs(plot_dir, exist_ok=True)  # Create folder if not exists


    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath)
    print(f"Plot saved as {filename}")


    plt.tight_layout()
    plt.show()