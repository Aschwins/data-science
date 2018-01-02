---
title: "Polynomial Regression"
date: 2017-09-15T11:45:36+02:00
categories:
  - Machine Learning
tags:
  - Python
  - R
  - Regression
---

---

# Polynomial Regression

In statistics, polynomial regression is a form of regression analysis in which the relationship between the independent variable X and the dependent variable y is modelled as an nth degree polynomial in X. Polynomial regression fits a nonlinear relationship between the value of X and the corresponding conditional mean of y, denoted E(y |X), and has been used to describe nonlinear phenomena such as the growth rate of tissues, the distribution of carbon isotopes in lake sediments, and the progression of disease epidemics.

## Introduction

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
# Polynomial Regression

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
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
~~~

Produces two images:

![Truth or Bluff (Linear Regression)](/images/polypy1.png)

As you can see the linear regression model we learned about before is not really suitable for our salary problem this time. The CEO making to much dough. So we have to make new model. Which we did in the code above aswell. Which produces:

![Truth or Bluff (Polynomial Regression)](/images/polypy2.png)

It fits the model, but lets smooth things over by making the resolution a bit higher. Sweet curves no?

![Truth or Bluff (Smoothed)](/images/polypy3.png)

In the second and third model, which are fitted pretty well, we can see that someone with a 6,5 position level makes around 200k. So this guy is problably not bluffing...

## In R

~~~
# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
#library(caTools)

#set.seed(123)
#split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
#training_set = subset(dataset, split == TRUE)
#test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

# Fitting Polynomial Regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,data = dataset)


#Plotting the linear regressor
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Salary vs Level (Linear Regression)') +
  xlab('Level') +
  ylab('Salary')

#Plotting the polynomial regressor
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Salary vs Level (Polynomia; Regression)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(poly_reg,
                                        newdata = data.frame(Level = x_grid,
                                                             Level2 = x_grid^2,
                                                             Level3 = x_grid^3,
                                                             Level4 = x_grid^4))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

#Predicting a new result with linear Regression
y_pred = predict(lin_reg, newdata = data.frame(Level=6.5))

#Predicting a new result with linear Regression
y_pred = predict(poly_reg, newdata = data.frame(Level=6.5,
                                                Level2 = 6.5^2,
                                                Level3 = 6.5^3,
                                                Level4 = 6.5^4))
  ~~~

Produces the slightly prettier images, with somewhat weird titles. Please ignore, lazy.

![Truth or Bluff?](/images/polyr1.png)

Same story as before. Linear Regression not sufficient.

![Truth or Bluff?](/images/polyr2.png)

Smoothing this over:

![Truth or Bluff?](/images/polyr3.png)

This guy is for real, probably...

# A small Polynomial regression notebook.


# Polynomial Regression

What if your data doesn't look linear at all? Let's look at some more realistic-looking page speed / purchase data:


```python
%matplotlib inline
from pylab import *
import numpy as np

np.random.seed(2)
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

scatter(pageSpeeds, purchaseAmount)
```




    <matplotlib.collections.PathCollection at 0x16cb83bd978>




![png](/images/output_2_1 (2).png)


numpy has a handy polyfit function we can use, to let us construct an nth-degree polynomial model of our data that minimizes squared error. Let's try it with a 4th degree polynomial:


```python
x = np.array(pageSpeeds)
y = np.array(purchaseAmount)

p4 = np.poly1d(np.polyfit(x, y, 4))

```

We'll visualize our original scatter plot, together with a plot of our predicted values using the polynomial for page speed times ranging from 0-7 seconds:


```python
import matplotlib.pyplot as plt

xp = np.linspace(0, 7, 100)
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()
```


![png](/images/output_6_0 (2).png)


Looks pretty good! Let's measure the r-squared error:


```python
from sklearn.metrics import r2_score

r2 = r2_score(y, p4(x))

print(r2)

```

    0.82937663963


## Activity

Try different polynomial orders. Can you get a better fit with higher orders? Do you start to see overfitting, even though the r-squared score looks good for this particular data set?


```python

```

# Playing around with Train/Test


# Train / Test

We'll start by creating some data set that we want to build a model for (in this case a polynomial regression):


```python
%matplotlib inline
import numpy as np
from pylab import *

np.random.seed(2)

pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeeds


scatter(pageSpeeds, purchaseAmount)
```




    <matplotlib.collections.PathCollection at 0x2a4ffd11b38>




![png](/images/output_2_1 (3).png)


Now we'll split the data in two - 80% of it will be used for "training" our model, and the other 20% for testing it. This way we can avoid overfitting.


```python
trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]

```

Here's our training dataset:


```python
scatter(trainX, trainY)
```




    <matplotlib.collections.PathCollection at 0x2a4ffdec2b0>




![png](/images/output_6_1.png)


And our test dataset:


```python
scatter(testX, testY)
```




    <matplotlib.collections.PathCollection at 0x2a4ffe96cf8>




![png](/images/output_8_1.png)


Now we'll try to fit an 8th-degree polynomial to this data (which is almost certainly overfitting, given what we know about how it was generated!)


```python
x = np.array(trainX)
y = np.array(trainY)

p4 = np.poly1d(np.polyfit(x, y, 3))
```

Let's plot our polynomial against the training data:


```python
import matplotlib.pyplot as plt

xp = np.linspace(0, 7, 100)
axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0, 200])
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()

```


![png](/images/output_12_0 (2).png)


And against our test data:


```python
testx = np.array(testX)
testy = np.array(testY)

axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0, 200])
plt.scatter(testx, testy)
plt.plot(xp, p4(xp), c='r')
plt.show()
```


![png](/images/output_14_0 (2).png)


Doesn't look that bad when you just eyeball it, but the r-squared score on the test data is kind of horrible! This tells us that our model isn't all that great...


```python
from sklearn.metrics import r2_score

r2 = r2_score(testy, p4(testx))

print(r2)

```

    0.272743114752


...even though it fits the training data better:


```python
from sklearn.metrics import r2_score

r2 = r2_score(np.array(trainY), p4(np.array(trainX)))

print(r2)
```

    0.429578320121


If you're working with a Pandas DataFrame (using tabular, labeled data,) scikit-learn has built-in train_test_split functions to make this easy to do.

Later we'll talk about even more robust forms of train/test, like K-fold cross-validation - where we try out multiple different splits of the data, to make sure we didn't just get lucky with where we split it.

## Activity

Try measuring the error on the test data using different degree polynomial fits. What degree works best?


```python

```
