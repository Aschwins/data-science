---
title: "Artificial Neural Networks"
date: 2017-10-20T11:45:36+02:00
categories:
  - Machine Learning
  - Deep Learning
tags:
  - Python
  - R
  - Artificial Neural Networks
---
For future learning!: https://www.udemy.com/deeplearning/?couponCode=YESDATA
---

Deep Learning is the most exciting and powerful branch of Machine Learning. Deep Learning models can be used for a variety of complex tasks:

* **Artificial Neural Networks for Regression and Classification**
* Convolutional Neural Networks for Computer Vision
* Recurrent Neural Networks for Time Series Analysis
* Self Organizing Maps for Feature Extraction
* Deep Boltzmann Machines for Recommendation Systems
* Auto Encoders for Recommendation Systems

In this part, you will understand and learn how to implement the following Deep Learning models:

* Artificial Neural Networks for a Business Problem
* Convolutional Neural Networks for a Computer Vision task

Geoffrey Hinton!

Efficient BackProp By Yann LeCun et al. (1998)

Plan of Attack

* The Neuron
* The Activation Function
* How do Neural Networks work? (example)
* How do Neural Networks learn?
* Gradient Descent
* Stochastic Gradient Descent
* Backpropagation

# The Neuron

[Neuron](https://upload.wikimedia.org/wikipedia/commons/3/30/Chemical_synapse_schema_cropped.jpg)

Which we will model like the following picture, so a computer can act like a human brain!

![Neuron - Computerized](/images/ann.png)

# The Activation Function

* Threshold Function
* Sigmoid Function: phi(x) = 1/(1+e^{-x})
* Rectifier Function: phi(x) = max(x,0)
* Hyperbolic Tangent Function

Activation functions work on the weight and explanatory variables in the Neural Network. In most cases Rectifier functions are being used in the Hidden Layer and Sigmoid Functions are being used in the output layer.

# How do Neural Networks work?

![how a neural network works](/images/howneuralnetworks.png)

# How do Neural Networks learn?

Cost functions, Backpropagation.

Additional Reading
A Neural Network in 13 lines of Python (Part 2 - Gradient Descent) Andrew Trask (2015)

Neural Networks and Deep Learning Michael Nielsen (2015)

# Artificial Neural Network

In this problem we are working for a bank. The bank has, like all banks, customers. These customers sometimes leave which is not good for the company. So we as data scientist are asked to predict wether a customer is leaving at the end of the month or not. We are only given a dataset of 10000 clients and for eacht client it is known if they left or not. Now lets make the classifier, no, the neural network classifier!

## The dataset Churn_Modelling

[Churn_Modelling](/data/Churn_Modelling.csv)

| RowNumber | CustomerId | Surname                 | CreditScore | Geography | Gender | Age | Tenure | Balance   | NumOfProducts | HasCrCard | IsActiveMember | EstimatedSalary | Exited |
|-----------|------------|-------------------------|-------------|-----------|--------|-----|--------|-----------|---------------|-----------|----------------|-----------------|--------|
| 1         | 15634602   | Hargrave                | 619         | France    | Female | 42  | 2      | 0         | 1             | 1         | 1              | 101348.88       | 1      |
| 2         | 15647311   | Hill                    | 608         | Spain     | Female | 41  | 1      | 83807.86  | 1             | 0         | 1              | 112542.58       | 0      |
| 3         | 15619304   | Onio                    | 502         | France    | Female | 42  | 8      | 159660.8  | 3             | 1         | 0              | 113931.57       | 1      |
| 4         | 15701354   | Boni                    | 699         | France    | Female | 39  | 1      | 0         | 2             | 0         | 0              | 93826.63        | 0      |
| 5         | 15737888   | Mitchell                | 850         | Spain     | Female | 43  | 2      | 125510.82 | 1             | 1         | 1              | 79084.1         | 0      |
| 6         | 15574012   | Chu                     | 645         | Spain     | Male   | 44  | 8      | 113755.78 | 2             | 1         | 0              | 149756.71       | 1      |
| 7         | 15592531   | Bartlett                | 822         | France    | Male   | 50  | 7      | 0         | 2             | 1         | 1              | 10062.8         | 0      |
| 8         | 15656148   | Obinna                  | 376         | Germany   | Female | 29  | 4      | 115046.74 | 4             | 1         | 0              | 119346.88       | 1      |
| 9         | 15792365   | He                      | 501         | France    | Male   | 44  | 4      | 142051.07 | 2             | 0         | 1              | 74940.5         | 0      |
| ...      | ...   | ...                | ...        | ...     | ...   | .  | .      | .         | .             | .         | .             | .....        | .      |
| 9990      | 15605622   | McMillan                | 841         | Spain     | Male   | 28  | 4      | 0         | 2             | 1         | 1              | 179436.6        | 0      |
| 9991      | 15798964   | Nkemakonam              | 714         | Germany   | Male   | 33  | 3      | 35016.6   | 1             | 1         | 0              | 53667.08        | 0      |
| 9992      | 15769959   | Ajuluchukwu             | 597         | France    | Female | 53  | 4      | 88381.21  | 1             | 1         | 0              | 69384.71        | 1      |
| 9993      | 15657105   | Chukwualuka             | 726         | Spain     | Male   | 36  | 2      | 0         | 1             | 1         | 0              | 195192.4        | 0      |
| 9994      | 15569266   | Rahman                  | 644         | France    | Male   | 28  | 7      | 155060.41 | 1             | 1         | 0              | 29179.52        | 0      |
| 9995      | 15719294   | Wood                    | 800         | France    | Female | 29  | 2      | 0         | 2             | 0         | 0              | 167773.55       | 0      |
| 9996      | 15606229   | Obijiaku                | 771         | France    | Male   | 39  | 5      | 0         | 2             | 1         | 0              | 96270.64        | 0      |
| 9997      | 15569892   | Johnstone               | 516         | France    | Male   | 35  | 10     | 57369.61  | 1             | 1         | 1              | 101699.77       | 0      |
| 9998      | 15584532   | Liu                     | 709         | France    | Female | 36  | 7      | 0         | 1             | 0         | 1              | 42085.58        | 1      |
| 9999      | 15682355   | Sabbatini               | 772         | Germany   | Male   | 42  | 3      | 75075.31  | 2             | 1         | 0              | 92888.52        | 1      |
| 10000     | 15628319   | Walker                  | 792         | France    | Female | 28  | 4      | 130142.79 | 1             | 1         | 0              | 38190.78        | 0      |

## Training the ANN

![Training the ANN](/images/trainingtheann.png)

## In Python

``` Python
# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```
Gives us a confusion matrix with around 85% accuracy.

``` Python
# Visualising the results
kansen = y_pred_n.flatten()
churn = y_test.astype('bool')
colors = np.repeat('', len(churn))
colors[:] = 'g'
colors[churn] = 'r'

plt.bar(np.arange(len(kansen)),kansen[np.argsort(kansen)], color=colors[np.argsort(kansen)])
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['figure.figsize'] = 15, 8
plt.title('Churn at a bank')
plt.xlabel('Client')
plt.ylabel('Chance')
plt.yticks([])
#plt.savefig('image1.png')
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['figure.figsize'] = 15, 8
plt.show()
```

Gives us the beautiful graph, which shows us our neural network works quite well.

![graph neural net](/images/image1.png)

## In R

in R

``` R
# ANN

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Encoding categorical data as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting the ANN to the Training set.
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Exited', training_frame = as.h2o(training_set),
                              activation = 'Rectifier',
                              hidden = c(6,6),
                              epochs = 100,
                              train_samples_per_iteration = 10)

# Predicting the Test set results
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = (prob_pred >0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

h2o.shutdown()
```
