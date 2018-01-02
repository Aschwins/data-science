---
title: "Linear Regression"
date: 2017-09-15T11:45:36+02:00
categories:
  - Machine Learning
tags:
  - Python
  - R
  - Regression
---

---

In statistics, linear regression is a linear approach for modeling the relationship between a scalar dependent variable y and one or more explanatory variables (or independent variables) denoted X. The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.

# Simple Linear Regression

In the following problem we are a data scientist for a company who's hiring a new employee. The employee has a certain amount of experience. We are asked to make a model for the salary of our surrent employees, so we can make the new employee a fair offer.

## The dataset Position_Salaries

[Salary Data](/data/Salary_Data.csv)

| YearsExperience | Salary    |
|-----------------|-----------|
| 1.1             | 39343.00  |
| 1.3             | 46205.00  |
| 1.5             | 37731.00  |
| 2.0             | 43525.00  |
| 2.2             | 39891.00  |
| 2.9             | 56642.00  |
| 3.0             | 60150.00  |
| 3.2             | 54445.00  |
| 3.2             | 64445.00  |
| 3.7             | 57189.00  |
| 3.9             | 63218.00  |
| 4.0             | 55794.00  |
| 4.0             | 56957.00  |
| 4.1             | 57081.00  |
| 4.5             | 61111.00  |
| 4.9             | 67938.00  |
| 5.1             | 66029.00  |
| 5.3             | 83088.00  |
| 5.9             | 81363.00  |
| 6.0             | 93940.00  |
| 6.8             | 91738.00  |
| 7.1             | 98273.00  |
| 7.9             | 101302.00 |
| 8.2             | 113812.00 |
| 8.7             | 109431.00 |
| 9.0             | 105582.00 |
| 9.5             | 116969.00 |
| 9.6             | 112635.00 |
| 10.3            | 122391.00 |
| 10.5            | 121872.00 |



## In Python

~~~
# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
~~~

Produces two images:

![Salary vs Experience (Training Set)](/images/salvsexptrain.png)

where the linear regression model (blue line) is fitted to the training set (red dots). We can check the performance of the model by fitting our linear regression model to the test set:

![Salary vs Experience (Test Set)](/images/salvsexptest.png)


## In R

~~~
# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Visualising the Training set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')

# Visualising the Test set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')
  ~~~

Produces the slightly prettier images:

![Salary vs Experience (Training Set)](/images/salvsexptrainR.png)

where the linear regression model (blue line) is fitted to the training set (red dots). We can check the performance of the model by fitting our linear regression model to the test set:

![Salary vs Experience (Test Set)](/images/salvsexptestR.png)

# A simple linear regression notebook.


# Linear Regression

Let's fabricate some data that shows a roughly linear relationship between page speed and amount purchased:


```python
%matplotlib inline
import numpy as np
from pylab import *

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 0.1, 1000)) * 3

scatter(pageSpeeds, purchaseAmount)
```




    <matplotlib.collections.PathCollection at 0x20e3126c940>




![png](/images/output_2_1.png)


As we only have two features, we can keep it simple and just use scipy.state.linregress:


```python
from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)

```

Not surprisngly, our R-squared value shows a really good fit:


```python
r_value ** 2
```




    0.98933636422743532



Let's use the slope and intercept we got from the regression to plot predicted values vs. observed:


```python
import matplotlib.pyplot as plt

def predict(x):
    return slope * x + intercept

fitLine = predict(pageSpeeds)

plt.scatter(pageSpeeds, purchaseAmount)
plt.plot(pageSpeeds, fitLine, c='r')
plt.show()
```


![png](/images/output_8_0 (2).png)


## Activity

Try increasing the random variation in the test data, and see what effect it has on the r-squared error value.

# Multiple Linear regression

Multiple linear regression attempts to model the relationship between two or more explanatory variables and a response variable by fitting a linear equation to observed data. Every value of the independent variable $x$ is associated with a value of the dependent variable $y$.

By still using least squares we end up with coefficients for each factor:
$$y = a_0 + a_1\cdot x_1 + a_2\cdot x_2 + ... + a_n\cdot x_n$$
for dependent variable $y$ and indepent variables $x_1,..., x_n$. The coefficients $a_0,...,a_n$ imply how important each factor is (if the data is all normalized!). Get rid of the ones that don't matter, measure the fit with r-squared and be aware that the different factors are not themselves dependent on each other!

## Introduction

For this problem we are helping an investor make a model for the succes rate of startups. We have a few explanatory variables and are asked to make a model to predict the startups profit. Investing in startups with high profits is considered wise.

## The dataset 50_Startups

Download the dataset: [Salary Data](/data/50_Startups.csv)

| R&D Spend | Administration | Marketing Spend | State      | Profit    |
|-----------|----------------|-----------------|------------|-----------|
| 165349.2  | 136897.8       | 471784.1        | New York   | 192261.83 |
| 162597.7  | 151377.59      | 443898.53       | California | 191792.06 |
| 153441.51 | 101145.55      | 407934.54       | Florida    | 191050.39 |
| 144372.41 | 118671.85      | 383199.62       | New York   | 182901.99 |
| 142107.34 | 91391.77       | 366168.42       | Florida    | 166187.94 |
| 131876.9  | 99814.71       | 362861.36       | New York   | 156991.12 |
| 134615.46 | 147198.87      | 127716.82       | California | 156122.51 |
| 130298.13 | 145530.06      | 323876.68       | Florida    | 155752.6  |
| 120542.52 | 148718.95      | 311613.29       | New York   | 152211.77 |
| 123334.88 | 108679.17      | 304981.62       | California | 149759.96 |
| 101913.08 | 110594.11      | 229160.95       | Florida    | 146121.95 |
| 100671.96 | 91790.61       | 249744.55       | California | 144259.4  |
| 93863.75  | 127320.38      | 249839.44       | Florida    | 141585.52 |
| 91992.39  | 135495.07      | 252664.93       | California | 134307.35 |
| 119943.24 | 156547.42      | 256512.92       | Florida    | 132602.65 |
| 114523.61 | 122616.84      | 261776.23       | New York   | 129917.04 |
| 78013.11  | 121597.55      | 264346.06       | California | 126992.93 |
| 94657.16  | 145077.58      | 282574.31       | New York   | 125370.37 |
| 91749.16  | 114175.79      | 294919.57       | Florida    | 124266.9  |
| 86419.7   | 153514.11      | 0               | New York   | 122776.86 |
| 76253.86  | 113867.3       | 298664.47       | California | 118474.03 |
| 78389.47  | 153773.43      | 299737.29       | New York   | 111313.02 |
| 73994.56  | 122782.75      | 303319.26       | Florida    | 110352.25 |
| 67532.53  | 105751.03      | 304768.73       | Florida    | 108733.99 |
| 77044.01  | 99281.34       | 140574.81       | New York   | 108552.04 |
| 64664.71  | 139553.16      | 137962.62       | California | 107404.34 |
| 75328.87  | 144135.98      | 134050.07       | Florida    | 105733.54 |
| 72107.6   | 127864.55      | 353183.81       | New York   | 105008.31 |
| 66051.52  | 182645.56      | 118148.2        | Florida    | 103282.38 |
| 65605.48  | 153032.06      | 107138.38       | New York   | 101004.64 |
| 61994.48  | 115641.28      | 91131.24        | Florida    | 99937.59  |
| 61136.38  | 152701.92      | 88218.23        | New York   | 97483.56  |
| 63408.86  | 129219.61      | 46085.25        | California | 97427.84  |
| 55493.95  | 103057.49      | 214634.81       | Florida    | 96778.92  |
| 46426.07  | 157693.92      | 210797.67       | California | 96712.8   |
| 46014.02  | 85047.44       | 205517.64       | New York   | 96479.51  |
| 28663.76  | 127056.21      | 201126.82       | Florida    | 90708.19  |
| 44069.95  | 51283.14       | 197029.42       | California | 89949.14  |
| 20229.59  | 65947.93       | 185265.1        | New York   | 81229.06  |
| 38558.51  | 82982.09       | 174999.3        | California | 81005.76  |
| 28754.33  | 118546.05      | 172795.67       | California | 78239.91  |
| 27892.92  | 84710.77       | 164470.71       | Florida    | 77798.83  |
| 23640.93  | 96189.63       | 148001.11       | California | 71498.49  |
| 15505.73  | 127382.3       | 35534.17        | New York   | 69758.98  |
| 22177.74  | 154806.14      | 28334.72        | California | 65200.33  |
| 1000.23   | 124153.04      | 1903.93         | New York   | 64926.08  |
| 1315.46   | 115816.21      | 297114.46       | Florida    | 49490.75  |
| 0         | 135426.92      | 0               | California | 42559.73  |
| 542.05    | 51743.15       | 0               | New York   | 35673.41  |
| 0         | 116983.8       | 45173.06        | California | 14681.4   |


## In Python

~~~
# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1) #adding b0 arg in mlr model

X_opt = X[:, [0,1,2,3,4,5]]
SL = 0.05
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
~~~

We don't produce any images because, multiple dimensions are kind of hard to plot. But we do get a table as an output in the command line which consists of a lot of valuable information about our model:

~~~
Out[4]:

<class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.947
Model:                            OLS   Adj. R-squared:                  0.945
Method:                 Least Squares   F-statistic:                     849.8
Date:                Fri, 15 Sep 2017   Prob (F-statistic):           3.50e-32
Time:                        16:36:13   Log-Likelihood:                -527.44
No. Observations:                  50   AIC:                             1059.
Df Residuals:                      48   BIC:                             1063.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.903e+04   2537.897     19.320      0.000    4.39e+04    5.41e+04
x1             0.8543      0.029     29.151      0.000       0.795       0.913
==============================================================================
Omnibus:                       13.727   Durbin-Watson:                   1.116
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               18.536
Skew:                          -0.911   Prob(JB):                     9.44e-05
Kurtosis:                       5.361   Cond. No.                     1.65e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.65e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
~~~

We conclude that the only explanatory variable that is important enough for out model is R&D spend. Later we will build better models, and we will see some different results.

# In R

~~~
# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Profit ~ .,
               data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

#Building the optimal model using Backward Elimination
regressor_opt = lm(formula = Profit ~ R.D.Spend + Administration +
                     Marketing.Spend + State, data=dataset)
summary(regressor_opt)

regressor_opt = lm(formula = Profit ~ R.D.Spend, data=dataset)
~~~

Produces a similar table. Where we can conclude what explanatory variables are of importance and which are not.

~~~
Call:
lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend +
    State, data = dataset)

Residuals:
   Min     1Q Median     3Q    Max
-33504  -4736     90   6672  17338

Coefficients:
                  Estimate Std. Error t value Pr(>|t|)    
(Intercept)      5.008e+04  6.953e+03   7.204 5.76e-09 ***
R.D.Spend        8.060e-01  4.641e-02  17.369  < 2e-16 ***
Administration  -2.700e-02  5.223e-02  -0.517    0.608    
Marketing.Spend  2.698e-02  1.714e-02   1.574    0.123    
State2           4.189e+01  3.256e+03   0.013    0.990    
State3           2.407e+02  3.339e+03   0.072    0.943    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 9439 on 44 degrees of freedom
Multiple R-squared:  0.9508,	Adjusted R-squared:  0.9452
F-statistic: 169.9 on 5 and 44 DF,  p-value: < 2.2e-16
~~~

That's it for now for linear regression. Not very interesting let go on to Polynomial Regression.


# A small multivariate regression notebook

Let's grab a small little data set of Blue Book car values:


```python
import pandas as pd

df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')

```


```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
      <th>Mileage</th>
      <th>Make</th>
      <th>Model</th>
      <th>Trim</th>
      <th>Type</th>
      <th>Cylinder</th>
      <th>Liter</th>
      <th>Doors</th>
      <th>Cruise</th>
      <th>Sound</th>
      <th>Leather</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17314.103129</td>
      <td>8221</td>
      <td>Buick</td>
      <td>Century</td>
      <td>Sedan 4D</td>
      <td>Sedan</td>
      <td>6</td>
      <td>3.1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17542.036083</td>
      <td>9135</td>
      <td>Buick</td>
      <td>Century</td>
      <td>Sedan 4D</td>
      <td>Sedan</td>
      <td>6</td>
      <td>3.1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16218.847862</td>
      <td>13196</td>
      <td>Buick</td>
      <td>Century</td>
      <td>Sedan 4D</td>
      <td>Sedan</td>
      <td>6</td>
      <td>3.1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16336.913140</td>
      <td>16342</td>
      <td>Buick</td>
      <td>Century</td>
      <td>Sedan 4D</td>
      <td>Sedan</td>
      <td>6</td>
      <td>3.1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16339.170324</td>
      <td>19832</td>
      <td>Buick</td>
      <td>Century</td>
      <td>Sedan 4D</td>
      <td>Sedan</td>
      <td>6</td>
      <td>3.1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We can use pandas to split up this matrix into the feature vectors we're interested in, and the value we're trying to predict.

Note how we are avoiding the make and model; regressions don't work well with ordinal values, unless you can convert them into some numerical order that makes sense somehow.

Let's scale our feature data into the same range so we can easily compare the coefficients we end up with. After that we'll use OLS (=Ordinary Least Squares) from the statsmodels.api library to get the best estimations of our coefficients.


```python
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']

X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].as_matrix())

print (X)

est = sm.OLS(y, X).fit()

est.summary()
```

          Mileage  Cylinder     Doors
    0   -1.417485  0.527410  0.556279
    1   -1.305902  0.527410  0.556279
    2   -0.810128  0.527410  0.556279
    3   -0.426058  0.527410  0.556279
    4    0.000008  0.527410  0.556279
    5    0.293493  0.527410  0.556279
    6    0.335001  0.527410  0.556279
    7    0.382369  0.527410  0.556279
    8    0.511409  0.527410  0.556279
    9    0.914768  0.527410  0.556279
    10  -1.171368  0.527410  0.556279
    11  -0.581834  0.527410  0.556279
    12  -0.390532  0.527410  0.556279
    13  -0.003899  0.527410  0.556279
    14   0.430591  0.527410  0.556279
    15   0.480156  0.527410  0.556279
    16   0.509822  0.527410  0.556279
    17   0.757160  0.527410  0.556279
    18   1.594886  0.527410  0.556279
    19   1.810849  0.527410  0.556279
    20  -1.326046  0.527410  0.556279
    21  -1.129860  0.527410  0.556279
    22  -0.667658  0.527410  0.556279
    23  -0.405792  0.527410  0.556279
    24  -0.112796  0.527410  0.556279
    25  -0.044552  0.527410  0.556279
    26   0.190700  0.527410  0.556279
    27   0.337442  0.527410  0.556279
    28   0.566102  0.527410  0.556279
    29   0.660837  0.527410  0.556279
    ..        ...       ...       ...
    774 -0.161262 -0.914896  0.556279
    775 -0.089234 -0.914896  0.556279
    776 -0.040523 -0.914896  0.556279
    777  0.002572 -0.914896  0.556279
    778  0.236603 -0.914896  0.556279
    779  0.249666 -0.914896  0.556279
    780  0.357220 -0.914896  0.556279
    781  0.365521 -0.914896  0.556279
    782  0.434131 -0.914896  0.556279
    783  0.517269 -0.914896  0.556279
    784  0.589908 -0.914896  0.556279
    785  0.599186 -0.914896  0.556279
    786  0.793052 -0.914896  0.556279
    787  1.033554 -0.914896  0.556279
    788  1.045762 -0.914896  0.556279
    789  1.205567 -0.914896  0.556279
    790  1.541414 -0.914896  0.556279
    791  1.561070 -0.914896  0.556279
    792  1.725026 -0.914896  0.556279
    793  1.851502 -0.914896  0.556279
    794 -1.709871  0.527410  0.556279
    795 -1.474375  0.527410  0.556279
    796 -1.187849  0.527410  0.556279
    797 -1.079929  0.527410  0.556279
    798 -0.682430  0.527410  0.556279
    799 -0.439853  0.527410  0.556279
    800 -0.089966  0.527410  0.556279
    801  0.079605  0.527410  0.556279
    802  0.750446  0.527410  0.556279
    803  1.932565  0.527410  0.556279

    [804 rows x 3 columns]



<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Price</td>      <th>  R-squared:         </th> <td>   0.064</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.060</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   18.11</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 05 Dec 2017</td> <th>  Prob (F-statistic):</th> <td>2.23e-11</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:01:01</td>     <th>  Log-Likelihood:    </th> <td> -9207.1</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   804</td>      <th>  AIC:               </th> <td>1.842e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   801</td>      <th>  BIC:               </th> <td>1.843e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Mileage</th>  <td>-1272.3412</td> <td>  804.623</td> <td>   -1.581</td> <td> 0.114</td> <td>-2851.759</td> <td>  307.077</td>
</tr>
<tr>
  <th>Cylinder</th> <td> 5587.4472</td> <td>  804.509</td> <td>    6.945</td> <td> 0.000</td> <td> 4008.252</td> <td> 7166.642</td>
</tr>
<tr>
  <th>Doors</th>    <td>-1404.5513</td> <td>  804.275</td> <td>   -1.746</td> <td> 0.081</td> <td>-2983.288</td> <td>  174.185</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>157.913</td> <th>  Durbin-Watson:     </th> <td>   0.008</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 257.529</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.278</td>  <th>  Prob(JB):          </th> <td>1.20e-56</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.074</td>  <th>  Cond. No.          </th> <td>    1.03</td>
</tr>
</table>


The table of coefficients above gives us the values to plug into an equation of form:
    B0 + B1 * Mileage + B2 * model_ord + B3 * doors

In this example, it's pretty clear that the number of cylinders is more important than anything based on the coefficients.

Could we have figured that out earlier?

```python
y.groupby(df.Doors).mean()
```




    Doors
    2    23807.135520
    4    20580.670749
    Name: Price, dtype: float64



Surprisingly, more doors does not mean a higher price! (Maybe it implies a sport car in some cases?) So it's not surprising that it's pretty useless as a predictor here. This is a very small data set however, so we can't really read much meaning into it.

## Activity

Mess around with the fake input data, and see if you can create a measurable influence of number of doors on price. Have some fun with it - why stop at 4 doors?
