#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Data import
print("Accessing the data...")
shared_articles = pd.read_csv("shared_articles.csv")
user_interactions = pd.read_csv("users_interactions.csv")

# Replacing different events with their weights
event_weight = { "VIEW": 0.01, "LIKE":0.04, "COMMENT CREATED": 0.10, "FOLLOW":0.25, "BOOKMARK":1.00 }
user_interactions["eventType"].replace(event_weight , inplace=True)

# print(user_interactions.columns)
# print(shared_articles.columns)

print("Data Processing...")
user_inter = user_interactions.drop_duplicates(subset = ["contentId"])
shared = shared_articles.drop_duplicates(subset = ["contentId"] )

# Virality score table for all articles which were shared on the platform.
virality_score = user_interactions.groupby("contentId")[["eventType"]].sum().reset_index()
virality_score["eventType"] = virality_score["eventType"].astype(int)

# Table for Article ID, Virality_score of article and other article information.
score_and_info = pd.merge(virality_score[["contentId","eventType"]],
                          shared[["contentId", "contentType", "url", "title", "text", "lang"]] ,
                          left_on = "contentId", right_on = "contentId", how = "left")

# column for all the text information of the articles to be used for calculation text similarity.
score_and_info["sum_of_words"] = score_and_info["contentType"] + " " + score_and_info["title"] + " " + score_and_info["text"] + " " + score_and_info["lang"]

# print(score_and_info.head(5))

# Test train split
X_train, X_test, y_train, y_test = train_test_split(score_and_info[["contentId", "sum_of_words"]],
                                                    score_and_info[["eventType"]],
                                   test_size=0.25, random_state=42)

train  = pd.concat([X_train, y_train], axis=1).reset_index()
test = pd.concat([X_test, y_test] , axis =1).reset_index()

documents = []
numOfTrainDocs = len(train)
numOfTestDocs = len(test)

print("Converting text to feature vectors... ")
# Creating the document list for the word2vec vectorizer.
for tr_index, train_row in train.iterrows() :
    documents.append(train_row["sum_of_words"])

for index, test_row in test.iterrows() :
    documents.append(test_row["sum_of_words"])


# Count Vectorizer does text preprocessing, tokenizing and filtering of stopwords,
# then it builds a dictionary of features and transforms documents to feature vectors
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train.loc[:,"sum_of_words"])
# print(X_train_counts.shape)

#To avoid discrepancies between documents of different lengths (hence different length vectors)
# we use TF-IDF Transformer to divide the number of occurrences of each word in a document by
# the total number of words in the document: these new features are called tf for Term Frequencies.
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print(X_train_tfidf.shape)

print("Training the model... ")
X = X_train_tfidf  # training feature vector
y = np.asarray(y_train.loc[: ,"eventType"]) # train output
clf = LogisticRegression() #Used a Logistic Regression classifier
clf = clf.fit(X, y)

# Test vectors
X_new_counts = count_vect.transform(X_test.loc[:,"sum_of_words"])
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

print("Accuracy score for a 75:25 train-test split using word2Vec and Logistic Regression is: " , round( (accuracy_score(y_test.loc[:,"eventType"], predicted))*100,2) )
