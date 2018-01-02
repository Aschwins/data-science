---
title: "Random Forest Regression"
date: 2017-09-21T11:45:36+02:00
categories:
  - Machine Learning
tags:
  - Python
  - R
  - Regression
---

---

Introduction

# Random Forest Regression

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
# Random Forest Regression

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
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
~~~

Produces the following image.

![Truth or Bluff (SVR)](/images/rf1.png)



## In R
~~~
# Random Forest Regression

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

# Fitting the Random Forest Regression to the dataset
#install.packages('randomForest')

library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1],
                         y = dataset$Salary,
                         ntree = 500)

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Random Forest Regression)') +
  xlab('Level') +
  ylab('Salary')
  ~~~

  Produces somewhat prettier pictures

  ![Truth or Bluff (SVR)](/images/rf2.png)



  So we can see that this guy probably is not bluffing. But we can also see that this model is not that accurate since it doesn't catch the CEO. With experience level 10.
