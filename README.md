Problem
You’re leading the product team at a content reading platform such as Beedly. Over time you’ve collected some data about content and user interactions with the content. You now want to build a model that predicts virality of a given article.
The data collected is described below and can be downloaded from Kaggle: Content:
The file "shared_articles" contains information about the articles shared in the platform.
Each article has its sharing date (timestamp), the original url, title, content in plain text, the article language (Portuguese - pt or English - en) and information about the user who shared the article (author).
There are two possible event types at a given timestamp:
● CONTENT SHARED: The article was shared in the platform and is available for users.
● CONTENT REMOVED: The article was removed from the platform.
Interactions:
This file user_interactions contains logs of user interactions on shared articles. It can be joined to
shared_articles.csv by contentId column. The eventType values are:
● VIEW: The user has opened the article.
● LIKE: The user has liked the article.
● COMMENT CREATED: The user created a comment in the article.
● FOLLOW: The user chose to be notified on any new comment in the article.
● BOOKMARK: The user has bookmarked the article for easy return in the future.
Some internal analysis has revealed that the metric representing virality is described as follows: VIRALITY = 1* VIEW + 4*LIKE + 10*COMMENT + 25*FOLLOW + 100*BOOKMARK
You’re to build a model that can predict virality of a new article being posted on the platform so that the news feed product can your model to showcase new articles.

Virality Score predictor for text articles. 

I considered text similarity as a feature for predicting the virality score. This a broad problem statement and a lot of features can be used for modelling for prediction of the virality score. In the submitted model I chose to calculate the virality of a new article (test set article) based on its content’s text similarity to the articles which are already on the platform (training set). 

Model description: 
Input: New Article’s (content + title + type + language) text information.  
o	The text is converted in a feature vector using the Count Vectorizer and TF-IDF Vectorizer. 
o	The Vectors are used to train a Logistic Regression model. 
o	Based on the feature vector information of the training set articles the virality score for the test article is predicted. 
Output: Virality score based on the given formula for the new article. 

I used word-to-vector plus logistic regression model because: 
o	I decided to use the text information to calculate the virality score assuming that is a criterion for virality. 
o	I chose Count Vectorizer and TF_IDF vectorizer to convert the text to feature vectors since these two together do text processing, removal of stop words, calculation of the frequencies of words in the document. This helps in understanding the text similarity irrespective of the differences in lengths of the document. 
o	I used a simple logistic regression model to train the system. This simplifies the model assuming discrete valued output. 
