---
title: "Natural Language Processing"
date: 2017-09-24T11:45:36+02:00
categories:
  - Machine Learning
tags:
  - Python
  - R
  - Natural Langue Processing (NLP)
---

---

Natural language processing (NLP) is a field of computer science, artificial intelligence and computational linguistics concerned with the interactions between computers and human (natural) languages, and, in particular, concerned with programming computers to fruitfully process large natural language corpora. Challenges in natural language processing frequently involve natural language understanding, natural language generation (frequently from formal, machine-readable logical forms), connecting language and machine perception, dialog systems, or some combination thereof.


# Natural Language Processing

In this problem we are working for a restaurant. This restaurant has been given a lot of reviews on YELP. We as data scientists are asked to analyse these reviews and make a model for the future where the restaurant can see if they get positive or negative reviews. In total we have a 1000 reviews. We could go through these manually or take the easy way out and let a machine learn how to interpret reviews!

## The dataset Restaurant_Reviews

[Restaurant_Reviews](/data/Restaurant_Reviews.tsv)

Since it is a dataset containing 1000 reviews we only show the top and bottom part of the Restaurant_Reviews. Also, it's important to notice that this is the first time we're not working with a csv file. This time we're working with a tsv, tab seperated value. Because reviews can contain comma's this is very wise. The reviews cannot contain tabs because whenever you press tab on website you go to the next entry. So we'll have to specify this in our code!

| Review                                                                                                                                                | Liked |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| Wow... Loved this place.                                                                                                                              | 1     |
| Crust is not good.                                                                                                                                    | 0     |
| Not tasty and the texture was just nasty.                                                                                                             | 0     |
| Stopped by during the late May bank holiday off Rick Steve recommendation and loved it.                                                               | 1     |
| The selection on the menu was great and so were the prices.                                                                                           | 1     |
| Now I am getting angry and I want my damn pho.                                                                                                        | 0     |
| Honeslty it didn't taste THAT fresh.)                                                                                                                 | 0     |
| The potatoes were like rubber and you could tell they had been made up ahead of time being kept under a warmer.                                       | 0     |
| The fries were great too.                                                                                                                             | 1     |
| A great touch.                                                                                                                                        | 1     |
|                                                 ...................................................                                                   | ....  |
| The refried beans that came with my meal were dried out and crusty and the food was bland.                                                            | 0     |
| Spend your money and time some place else.                                                                                                            | 0     |
| A lady at the table next to us found a live green caterpillar In her salad.                                                                           | 0     |
| the presentation of the food was awful.                                                                                                               | 0     |
| I can't tell you how disappointed I was.                                                                                                              | 0     |
| I think food should have flavor and texture and both were lacking.                                                                                    | 0     |
| Appetite instantly gone.                                                                                                                              | 0     |
| Overall I was not impressed and would not go back.                                                                                                    | 0     |
| The whole experience was underwhelming, and I think we'll just go to Ninja Sushi next time.                                                           | 0     |
| Then, as if I hadn't wasted enough of my life there, they poured salt in the wound by drawing out the time it took to bring the check.                | 0     |


Where 1 means a positive review and 0 means a negative one.

## In Python

First we have to clean the dataset and text before we can analyse it. In this case we'll only look at letters not numbers. We'll not look at capital letters, so lowercase all the way! Next we will only look at the stem of a word, so love=loves=loved. This way we can create a (not so huge) bad of words model.

~~~
# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
~~~

Notice, we only use the Naive Bayes classifier in the above example. We could use other classification models to test which are the best for NLP. We use:

``` Python
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train, y_train)

# Predicting the Test set results
y_pred_NB = NB.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_NB = confusion_matrix(y_test, y_pred_NB)

TP_NB = cm_NB[1,1]
TN_NB = cm_NB[0,0]
FP_NB = cm_NB[0,1]
FN_NB = cm_NB[1,0]

Accuracy_NB = (TP_NB + TN_NB) / (TP_NB + TN_NB + FP_NB + FN_NB)
Precision_NB = TP_NB / (TP_NB + FP_NB)
Recall_NB = TP_NB / (TP_NB+FN_NB)
F1_Score_NB = 2 * Precision_NB * Recall_NB / (Precision_NB + Recall_NB)

#------------------------ Random Forest Classification-----------------

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, y_train)

# Predicting the Test set results
y_pred_RF = RF.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RF = confusion_matrix(y_test, y_pred_RF)

TP_RF = cm_RF[1,1]
TN_RF = cm_RF[0,0]
FP_RF = cm_RF[0,1]
FN_RF = cm_RF[1,0]

Accuracy_RF = (TP_RF + TN_RF) / (TP_RF + TN_RF + FP_RF + FN_RF)
Precision_RF = TP_RF / (TP_RF + FP_RF)
Recall_RF = TP_RF / (TP_RF + FN_RF)
F1_Score_RF = 2 * Precision_RF * Recall_RF / (Precision_RF + Recall_RF)
```

Produces Confusion Matrix:

~~~
array([[55, 42],
       [12, 91]])
~~~

Evaluate the performance of each of these models. Try to beat the Accuracy obtained in Naive Bayes. But remember, Accuracy is not enough, so you should also look at other performance metrics like Precision (measuring exactness), Recall (measuring completeness) and the F1 Score (compromise between Precision and Recall). Please find below these metrics formulas (TP = # True Positives, TN = # True Negatives, FP = # False Positives, FN = # False Negatives):

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = 2 * Precision * Recall / (Precision + Recall)

Results:

| Classification model  | TP  | TN  | FP  | FN  | Accuracy  | Precision | Recall  | F1_score  |
| NB                    | 55  | 91  | 42  | 12  | 0.73      | 0.68      | 0.88    | 0.77      |
| RF                    | 57  | 87  | 10  | 46  | 0.72      | 0.85      | 0.55    | 0.67      |

A 73 percent precision rate is not to bad for a dataset containing only 1000 entry's.

## In R

NLP

``` Python
# Natural Language Processing

# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
```

If you want to check if the lines of code are working one could run `as.character(corpus[[841]])`, which gives:

~~~
> as.character(corpus[[841]])
[1] "buck head realli expect better food"
~~~
