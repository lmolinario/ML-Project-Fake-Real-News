import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def plot_news_data(ds_true, ds_fake, ds_news):
    """
    This funciton takes as input three datasets, one containing just the true news, one the false news and one a mix of
    the two. Plots various information: Most common words in each class, number of subjects in each class and the
    proportion fo fake and true news in the set of mixed news.
    Args:
        ds_true: set of true news
        ds_fake: set of fake news
        ds_news: set composed of both true and fake news
    """
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

    plt.show()

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

    # Display the plots
    plt.tight_layout()
    plt.show()

    # Count-plot of label values
    plt.figure(figsize=(8, 8))
    plt.title("Label Count")
    sns.countplot(x="label", data=ds_news)
    plt.show()

