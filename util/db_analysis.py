def get_unique_word_count(ds_news):
    """
    This function calculates the total number of unique words in the dataset.
    Args:
    ds_news (list): List of documents, where each document is a list of words.

    Returns:
    int: Total number of unique words in the dataset.
    """
    wordlist = []
    for doc in ds_news:
        wordlist.extend(doc)
    total_words = len(set(wordlist))
    return total_words


def get_max_min_word_count(ds_news):
    """
    This function calculates the maximum and minimum number of words in any document in the dataset.
    Args:
    ds_news (list): List of documents, where each document is a list of words.

    Returns:
    tuple: Maximum number of words, index of the document with the maximum number of words,
           minimum number of words, index of the document with the minimum number of words.
    """
    maxlen = -1
    minlen = float('inf')
    imax = imin = 0

    for i, doc in enumerate(ds_news):
        doc_len = len(doc)
        if doc_len > maxlen:
            maxlen = doc_len
            imax = i
        if doc_len < minlen:
            minlen = doc_len
            imin = i

    return maxlen, imax, minlen, imin


def get_max_min_unique_word_count(ds_news_unique):
    """
    This function calculates the maximum and minimum number of unique words in any document in the dataset.
    Args:
    ds_news_unique (list): List of documents, where each document is a set of unique words.

    Returns:
    tuple: Maximum number of unique words, index of the document with the maximum number of unique words,
           minimum number of unique words, index of the document with the minimum number of unique words.
    """
    maxdim = -1
    mindim = float('inf')
    imaxu = iminu = 0

    for i, doc in enumerate(ds_news_unique):
        unique_length = len(doc)
        if unique_length > maxdim:
            maxdim = unique_length
            imaxu = i
        if unique_length < mindim:
            mindim = unique_length
            iminu = i

    return maxdim, imaxu, mindim, iminu