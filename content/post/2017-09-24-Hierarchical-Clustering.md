---
title: "Hierarchical Clustering"
date: 2017-09-24T11:45:36+02:00
categories:
  - Machine Learning
tags:
  - Python
  - R
  - Clustering
---

---

In data mining and statistics, hierarchical clustering (also called hierarchical cluster analysis or HCA) is a method of cluster analysis which seeks to build a hierarchy of clusters. Strategies for hierarchical clustering generally fall into two types:[1]
Agglomerative: This is a "bottom up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
Divisive: This is a "top down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.
In general, the merges and splits are determined in a greedy manner. The results of hierarchical clustering are usually presented in a dendrogram.

# Hierarchical clustering

In the following problem we are working for a shopping mall. The mall has implemented a shopping system which uses cards, so customers can get discounts, and the mall is able to collect data.

From this data they have assigned a shopping score to all the customers. This scores is based on how much they spend in the mall.
We as a data scientist are asked to find out who our ideal customers are, so they can target them with their marketing and empty their pockets.
Since we're targeting, or searching out, specific customers this is clearly a clustering problem. Data science for the win.

## The dataset Mall_Customers

[Mall_Customers](/data/Mall_Customers.csv)

Since it is a dataset containing 200 entry's we only show the top and bottem part of the Mall_Customers.csv.

| CustomerID | Genre  | Age | Annual Income (k$) | Spending Score (1-100) |
|------------|--------|-----|--------------------|------------------------|
| 0001       | Male   | 19  | 15                 | 39                     |
| 0002       | Male   | 21  | 15                 | 81                     |
| 0003       | Female | 20  | 16                 | 6                      |
| 0004       | Female | 23  | 16                 | 77                     |
| 0005       | Female | 31  | 17                 | 40                     |
| 0006       | Female | 22  | 17                 | 76                     |
| 0007       | Female | 35  | 18                 | 6                      |
| 0008       | Female | 23  | 18                 | 94                     |
| 0009       | Male   | 64  | 19                 | 3                      |
| 0010       | Female | 30  | 19                 | 72                     |
|............|........|.....|....................| .......................|
| 0191       | Female | 34  | 103                | 23                     |
| 0192       | Female | 32  | 103                | 69                     |
| 0193       | Male   | 33  | 113                | 8                      |
| 0194       | Female | 38  | 113                | 91                     |
| 0195       | Female | 47  | 120                | 16                     |
| 0196       | Female | 35  | 120                | 79                     |
| 0197       | Female | 45  | 126                | 28                     |
| 0198       | Male   | 32  | 126                | 74                     |
| 0199       | Male   | 32  | 137                | 18                     |
| 0200       | Male   | 30  | 137                | 83                     |

## In Python

~~~
# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

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

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
~~~

In a clustering problem it always wise to determine how many clusters you should use for your model. We could put everyone in the same cluster or everyone in there own cluster. The optimal number of clusters is somewhere in between. Finding the optimal number of clusters can be done in several ways, for hierarchical clustering one could use a dendrogram.

![Hierarchical (Dendrogram)](/images/hc1.png)

To get the optimal number of clusters you just have to remember one simple trick: Find the biggest difference in euclidean distance of two different amounts of clusters. In the above graph you can find this by noticing the line between 100 and 245 is the longest line which doesn't get cut of by making a new cluster. Intersecting by drawing a horizontal line just in the middle of it gives us 5 intersections with vertical lines, so the optimal number of clusters is!?

The dendrogram gives us the optimal number of clusters, 5! After applying hc to our dataset with 5 clusters, plotting the clusters, and giving the clusters the appropriate names, we get.

![Hierarchical clustering](/images/hc2.png)

Notice we're missing centroids because this method doesn't use centroids. Just euclidean distance.

## In R

~~~
# Hierarchical Clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
dataset = dataset[4:5]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Using the dendrogram to find the optimal number of clusters
dendrogram = hclust(d = dist(dataset, method = 'euclidean'), method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

# Fitting Hierarchical Clustering to the dataset
hc = hclust(d = dist(dataset, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

# Visualising the clusters
library(cluster)
clusplot(dataset,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels= 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')
~~~

Produces somewhat worse pictures, but still correct.

![Hierarchical clustering(Dendogram)](/images/hc3.png)

![Hierarchical clustering](/images/hc4.png)

So the mall should target the top right cluster, and maybe the careless ones aswell. Those are our ideal customers.

Mo money mo problems tho.
