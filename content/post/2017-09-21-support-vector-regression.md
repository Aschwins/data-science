---
title: "Support Vector Regression"
date: 2017-09-21T11:45:36+02:00
categories:
  - Machine Learning
tags:
  - Python
  - R
  - Regression
---

---
In machine learning, support vector machines (SVMs, also support vector networks[1]) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.

# Support Vector Machines

In the following problem we are a data scientist for a company who is again hiring! Man I love new collegues. So this guy walks in and he says he has been a regional manager for quite some time. Pretty bad ass right? Now he's telling us he's making 160k a year at his last company!

What do you think, this guy for real?

Lets data science the shit out of this bitch to see if he's bluffing!

## The dataset Position_Salaries

[Salary Data](/data/Position_Salaries.csv)

| Position          | Level | Salary  |
|-------------------|-------|---------|
| Business Analyst  | 1     | 45000   |
| Junior Consultant | 2     | 50000   |
| Senior Consultant | 3     | 60000   |
| Manager           | 4     | 80000   |
| Country Manager   | 5     | 110000  |
| Region Manager    | 6     | 150000  |
| Partner           | 7     | 200000  |
| Senior Partner    | 8     | 300000  |
| C-level           | 9     | 500000  |
| CEO               | 10    | 1000000 |

## In Python

~~~
# SVR Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Fitting the SVR Model to the dataset
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X,y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
~~~

Produces the following image.

![Truth or Bluff (SVR)](/images/svr1.png)

Smoothing things over produces:

![Truth or Bluff (SVR)](/images/svr2.png)

## In R

~~~
# SVR Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the SVR Model to the dataset
install.packages('e1071')
library(e1071)
regressor = svm(formula = Salary ~ .,
                data= dataset,
                type = 'eps-regression')

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the SVR Model results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the SVR Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +
  ylab('Salary')
  ~~~

  Produces somewhat prettier pictures

  ![Truth or Bluff (SVR)](/images/svrr1.png)

  Smoothing things over produces:

  ![Truth or Bluff (SVR)](/images/svrr2.png)

  So we can see that this guy probably is not bluffing. But we can also see that this model is not that accurate since it doesn't catch the CEO. With experience level 10.
