![Fake_Real_News_ML](./doc/fake_real_news_AI.jpeg "Fake and real news dataset | Kaggle")
# Fake and real news dataset | Kaggle
This project is developed for the UNICA.IT University ML exams. 

Starting from the datasets provided by "Kaggle.com" we have structured an algorithm capable of distinguishing fake news from real news.


> **Master's degree in Computer Engineering, Cybersecurity and Artificial Intelligence - University of Cagliari**

> **Machine Learning**

> **Authors**: Michela Lecca - Marco Mulas- Lello Molinario

> **Supervisor**:Prof. Battista Biggio 
***
***
# Contents
1. [Installation](#installation)
2. [Project Goal](#project-goal)
3. [Solution Design](#solution-design)
4. [Analysis and conclusion](#Analysis-and-conclusion)

***
***

## Installation

- Download the ZIP code or clone the repository with:
  ```bash
  git clone https://github.com/lmolinario/MLproject.git
  ```
- Install the requirements with:

  ```bash
  pip3 install -r requirements.txt
  ```
- Run the file `entrypoint.py` to start the program.

## Project goal
The proposed project aims to deepen knowledge of the key concepts and potential of machine learning algorithms. 
Through binary classification the elements of a dataset are divided into two groups, managing to predict which group each element belongs to.

## Solution Design
The developed code implements a modular text classification pipeline using the strategy design pattern, allowing data representation and interchangeable classification strategies. The Strategy Design Pattern allows you to define a family of algorithms, encapsulating each one and making them interchangeable without altering the client code.

It includes three methods of data representation:
1. **`TokenizerRepresentation`**: Convert text to integer sequences using Keras Tokenizer and fixed-length pad sequences.
2. **`TextVectorizationRepresentation`**: Use TensorFlow's TextVectorization layer to map text into integer sequences and fill them.
3. **`TFIDFRepresentation`**: Applies TF-IDF vectorization to convert text into numeric features based on document frequency inverse to term frequency.
Using these methods, you transform text data into numeric formats suitable for machine learning models.

 The script also provides three classification strategies:
1. **`NaiveBayesClassifier`**: Implements Naive Bayes for binary classification,such as text classification tasks.
2. **`PerceptronNetStrategy`**:  A basic form of neural network, suitable for binary classification tasks.
3. **`RandomForestClassifier`**: Implements a Random Forest algorithm with multiple decision trees for robust classification.

The performance of the classifiers is evaluated through:
1. **`Average accuracy`**:Precision is the ratio of actual positive predictions to the total predicted positives. 
2. **`Average recall`**:Recall is the ratio of true positive predictions to total actual positives. 
3. **`Average F1 Score`**: The F1 score is the harmonic mean of precision and recall.

The pipeline supports k-fold cross-validation for robust evaluation of model performance.

## Analysis and conclusion
to implement...
