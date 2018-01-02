---
title: "Convolutional Neural Networks"
date: 2017-10-20T11:45:36+02:00
categories:
  - Machine Learning
  - Deep Learning
tags:
  - Python
  - R
  - Convolutional Neural Networks
---
For future learning!: https://www.udemy.com/deeplearning/?couponCode=YESDATA
---

Deep Learning is the most exciting and powerful branch of Machine Learning. Deep Learning models can be used for a variety of complex tasks:

* Artificial Neural Networks for Regression and Classification
* Convolutional Neural Networks for Computer Vision
* Recurrent Neural Networks for Time Series Analysis
* Self Organizing Maps for Feature Extraction
* Deep Boltzmann Machines for Recommendation Systems
* Auto Encoders for Recommendation Systems

In this part, you will understand and learn how to implement the following Deep Learning models:

* Artificial Neural Networks for a Business Problem
* **Convolutional Neural Networks for a Computer Vision task**

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

# Convolutional Neural Networks

Convolutional Neural Networks are able to proces images it has been given. It's basically learning a computer to see. Very cool stuff, but how does it actually work? Well the picture it has been given gets transformed so it can be translated to numbers. After this it gets thrown into a huge pile of linear algebra, and voila!

## Plan of Attack

* What are Convolutional Neural Networks
* Step 1 - Convolution Operation
* Step 1(b) - ReLU LayerS
* Step 2 - Pooling
* Step 3 - Flattening
* Step 4 - Full Connection
* Summary
* Softmax & Cross-Entropy

What do you see?

![Image recon](/images/cnn1.png)

Yann LeCun! Gradient-Based Learning Applied to Document Recognition

![How a CNN works](/images/cnn2.png)

* Step 1: Convolution
* Step 2: Max Pooling
* Step 3: Flattening
* Step 4: Full Connection

Jianxin Wu - Introduction to Convolutional Neural Networks

![How a CNN works](/images/cnn3.png)

Convolutional functions

![How a CNN works](/images/cnn4.png)

![How a CNN works](/images/cnn5.png)

Understanding Convolutional Neural Networks with A Mathematical Model, C. -C. Jay Kuo (2016)
https://arxiv.org/pdf/1609.04112.pdf

Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, Kaimin He
https://arxiv.org/pdf/1502.01852.pdf

## Max Pooling

Also called downsampling.

![How a CNN works](/images/cnn6.png)

Evaluation of Pooling
Operations in Convolutional Architectures for Object Recognition, Dominik Scherer
http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf

## Flattening

Flattening is transforming the pooled layer to one long vector.

## Full Connection

Add a Neural Network!

![cnn](/images/cnn7.png)

The 9 Deep Learning Papers You Need To Know About, Adit Deshpande (2016)
https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html

## Softmax & Cross Entropy

In mathematics, the softmax function, or normalized exponential function, is a generalization of the logistic function that "squashes" a K-dimensional vector of arbitrary real values to a K-dimensional vector of real values in the range [0, 1] that add up to 1. The function is given by

katex.render("c = \\pm\\sqrt{a^2 + b^2}", element);

In probability theory, the output of the softmax function can be used to represent a categorical distribution – that is, a probability distribution over K different possible outcomes. In fact, it is the gradient-log-normalizer of the categorical probability distribution.

Cross Entropy is just a lot better at recognising and improving accuracy than for example a mean square error method. Since it uses a log function it is able to detect smaller mistakes. Which helps the neural network to pick it up!

# Computer Vision

``` Python
# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# https://keras.io/ <- Check it out!

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# From keras.io, randaom image generator.
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)
```

Gives around 85% accuracy. Had to run for about 25 hours. Next step is to build an application which takes the input of one picture and uses the neural network to classify it. Basically an application which can work with new data.
