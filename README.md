Virality Score predictor for text articles. 

I considered text similarity as a feature for predicting the virality score. This a broad problem statement and a lot of features can be used for modelling for prediction of the virality score. In the submitted model I chose to calculate the virality of a new article (test set article) based on its content’s text similarity to the articles which are already on the platform (training set). 

-	What model did you use and why?
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
