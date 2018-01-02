---
title: "Naive Bayes Classification"
date: 2017-09-24T11:45:36+02:00
categories:
  - Machine Learning
tags:
  - Python
  - R
  - Classification
---

---

In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
Naive Bayes has been studied extensively since the 1950s. It was introduced under a different name into the text retrieval community in the early 1960s,[1]:488 and remains a popular (baseline) method for text categorization, the problem of judging documents as belonging to one category or the other (such as spam or legitimate, sports or politics, etc.) with word frequencies as the features. With appropriate pre-processing, it is competitive in this domain with more advanced methods including support vector machines.[2] It also finds application in automatic medical diagnosis.[3]

# Naive Bayes Classification

In the following problem we are a data scientist for a company who is selling a very nice car. An expensive SUV on sale! The company wants to advertise their sale, so they can get rid of all the cars and make a lot of money. They don't want to spend to much on their marketing, so we're asked to make a model to target people who are likely to buy the car after they've seen the ad. We got a data-set of a marketingcampaign for a similar project/car.



## The dataset Social Network Ads

[Social_Network_Ads](/data/Social_Network_Ads.csv)

Since it is a dataset containing 400 entry's we only show the top and bottem part of the Social Network ads.

| User ID  | Gender | Age | EstimatedSalary | Purchased |
|----------|--------|-----|-----------------|-----------|
| 15624510 | Male   | 19  | 19000           | 0         |
| 15810944 | Male   | 35  | 20000           | 0         |
| 15668575 | Female | 26  | 43000           | 0         |
| 15603246 | Female | 27  | 57000           | 0         |
| 15804002 | Male   | 19  | 76000           | 0         |
| 15728773 | Male   | 27  | 58000           | 0         |
| 15598044 | Female | 27  | 84000           | 0         |
| 15694829 | Female | 32  | 150000          | 1         |
| 15600575 | Male   | 25  | 33000           | 0         |
| 15727311 | Female | 35  | 65000           | 0         |
| 15570769 | Female | 26  | 80000           | 0         |
| 15606274 | Female | 26  | 52000           | 0         |
| 15746139 | Male   | 20  | 86000           | 0         |
| 15704987 | Male   | 32  | 18000           | 0         |
| 15628972 | Male   | 18  | 82000           | 0         |
| 15697686 | Male   | 29  | 80000           | 0         |
| 15733883 | Male   | 47  | 25000           | 1         |
| 15617482 | Male   | 45  | 26000           | 1         |
| 15704583 | Male   | 46  | 28000           | 1         |
| ........ |....... |.... |................ |.......... |
| 15706071 | Male   | 51  | 23000           | 1         |
| 15654296 | Female | 50  | 20000           | 1         |
| 15755018 | Male   | 36  | 33000           | 0         |
| 15594041 | Female | 49  | 36000           | 1         |

## In Python

~~~
# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
~~~

Training Set

![Naive Bayes (Training Set)](/images/nb1.png)

Test set:

![Naive Bayes (Test Set)](/images/nb2.png)

The Naive Bayes classifier produces a beautiful continuous boundary. The classifier classifies everything in the red region as: WILL NOT BUY, and everything in the green region as: WILL BUY. So the car company should target people in the green region for their ads.

In the above example, like in most machine learning problems, we've created our model on the training set and are checking it's performance on the test set. We can check it's performance by running: `cm = confusion_matrix(y_test, y_pred)`. This gives us the confusion matrix with all the mistakes, false positves and false negatives.

## In R

~~~
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Naive Bayes (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Naive Bayes (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
~~~

Produces somewhat prettier pictures

![Naive Bayes (Training set)](/images/nb3.png)

![Naive Bayes (Test set)](/images/nb4.png)

The car company should target people in the green area for their marketing campaign.

# Naive Bayes notebook

Another good example of a Naive Bayes classification application is a spam classifier for e-mails. The way it uses Bayes is theomerem is in the following way:

Let FREE = $\{E-mail contains the word Free\}$

$$P(SPAM|FREE) = \frac{P(FREE|SPAM)P(SPAM)}/{P(FREE)}$$

Where the numerator is the probability of a message being spam and containing the word free. Where the denominator is the overall probability of an email containing the word free.


# Naive Bayes (the easy way)

We'll cheat by using sklearn.naive_bayes to train a spam classifier! Most of the code is just loading our training data into a pandas DataFrame that we can play with:

In this problem we'll not be cleaning the dataset. The dataset still contains HTML code, bodies etc. If we would make a real e-mail classifier we would have to clean the dataset first, remove nonsense words, remove capitols, numbers etc. We will learn how to do this in NLP, although getting rid of all the HTML code must be done with Beautiful Soup.

```python
import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory('C:/Users/1asch/udemy-datascience/emails/spam', 'spam'))
data = data.append(dataFrameFromDirectory('C:/Users/1asch/udemy-datascience/emails/ham', 'ham'))

```

Let's have a look at that DataFrame:


```python
data.iloc[347][1]
```




    '<html>\n\n<head>\n\n</head>\n\n<body>\n\n<p align="center"><font style="font-size: 11pt" face="Courier">Volume 8, Issue 35 - Sept. 2002</font></p>\n\n<p align="center"><b><a href="http://www.globalcybercollective.com/BBAN_9_2002.htm">CLICK HERE</a></b></p>\n\n<p align="center"><img border="0" src="http://www.globalcybercollective.com/WSBulletinEmail5.bmp"></p>\n\n<p align="left">&nbsp;</p>\n\n<p align="left">&nbsp;</p>\n\n<p align="center"><font size="2">I no longer wish to receive your newsletter <a href="mailto:WSB20000@444.net?subject=takeoff"><b>click here</b></a></font></p>\n\n</body>\n\n</html>\n\n\n\nurfrdemubblkunmdbyh\n\n\n\n\n'



Now we will use a CountVectorizer to split up each message into its list of words, and throw that into a MultinomialNB classifier. Call fit() and we've got a trained spam filter ready to go! It's just that easy.


```python
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)
```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)



Let's try it out:


```python
examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
predictions
```




    array(['spam', 'ham'],
          dtype='<U4')



## Activity

Our data set is small, so our spam classifier isn't actually very good. Try running some different test emails through it and see if you get the results you expect.

If you really want to challenge yourself, try applying train/test to this spam classifier - see how well it can predict some subset of the ham and spam emails.


```python

```
