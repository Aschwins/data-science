---
title: "SVM Classification"
date: 2017-09-24T11:45:36+02:00
categories:
  - Machine Learning
tags:
  - Python
  - R
  - Classification
---

---

In machine learning, support vector machines (SVMs, also support vector networks[1]) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.

# SVM Classification

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
# Support Vector Machine (SVM)

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

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
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
plt.title('SVM (Training set)')
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
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
~~~

Training Set

![SVM (Training Set)](/images/svm1.png)

Test set:

![SVM (Test Set)](/images/svm2.png)

We can clearly see that SVM classifier is a linear classifier. The classifier classifies everything in the red region as: WILL NOT BUY, and everything in the green region as: WILL BUY. So the car company should target people in the green region for their ads.

In the above example, like in most machine learning problems, we've created our model on the training set and are checking it's performance on the test set. We can check it's performance by running: `cm = confusion_matrix(y_test, y_pred)`. This gives us the confusion matrix with all the mistakes, false positves and false negatives.

## In R

~~~
# Support Vector Machine (SVM)

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
  ~~~

  Produces somewhat prettier pictures

![SVM (Training set)](/images/svm3.png)

![SVM (Test set)](/images/svm4.png)

The car company should target people in the green area for their marketing campaign. Now let's see if a non linear model makes better predictions!

# A small SVC Notebook


# Support Vector Machines

Let's create the same fake income / age clustered data that we used for our K-Means clustering example:


```python
import numpy as np

#Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    pointsPerCluster = float(N)/k
    X = []
    y = []
    for i in range (k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y
```


```python
%matplotlib inline
from pylab import *

(X, y) = createClusteredData(100, 5)

plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
plt.show()
```


![png](/images/output_3_0.png)


Now we'll use linear SVC to partition our graph into clusters:


```python
from sklearn import svm, datasets

C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
```

By setting up a dense mesh of points in the grid and classifying all of them, we can render the regions of each cluster as distinct colors:


```python
def plotPredictions(clf):
    xx, yy = np.meshgrid(np.arange(0, 250000, 10),
                     np.arange(10, 70, 0.5))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(figsize=(8, 6))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
    plt.show()

plotPredictions(svc)
```


![png](/images/output_7_0.png)


Or just use predict for a given point:


```python
print(svc.predict([[200000, 40]]))
```

    [2]



```python
print(svc.predict([[50000, 65]]))
```

    [1]


## Activity

"Linear" is one of many kernels scikit-learn supports on SVC. Look up the documentation for scikit-learn online to find out what the other possible kernel options are. Do any of them work well for this data set?


```python

```
