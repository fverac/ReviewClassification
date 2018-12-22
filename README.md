# Review Classification via feature engineering

This repository contains code that takes reviews aggregated from Amazon, Yelp, and IMDb and engineers the plaintext data for classification in two ways.

The two methods used to prepare the data were Bag of Words and Word Embedding.

For Bag of Words, encoded data into a matrix where each row is a document and each column corresponds to a word in the vocabulary of the whole dataset. The value at the ith row and jth column would be the number of times word j showed up in document i. This was done using the sklearn.feature_extraction.text package, specifically CountVectorizer.

For Word Embedding, used GLoVe word embedding (where a word is algorithmically converted to  a vector in a way that preserves some semantic relations) to construct a matrix where each row corresponds to a document and each column is a dimension of the word embedding. The value at the ith row and jth column would be the sum of the jth feature in the word embedding for all words in document i.


## Installation

Will have to install scikit-learn and matplot packages.

Will also have to download the glove 6b word embedding [here](https://nlp.stanford.edu/projects/glove/) and place the txt file in the same directory as reviews.py


## Python files

reviews.py : contains the code for both bag of words and word embeddings. In its current state it runs GridSearchCV for a logistic regression classifier on the word embeddings data. It also plots the results of GridSearchCV for both training and validation accuracies across all of the parameters.


## Data files
trainreviews.txt : the files of reviews, each line is a review with a tab seperating the review and the actual sentiment. 0 for bad, 1 for good.

testreviewsunlabeled.txt : same as trainreviews.txt, only actual sentiments are missing

