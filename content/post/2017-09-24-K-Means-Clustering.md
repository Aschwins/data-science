---
title: "K-Means Clustering"
date: 2017-09-24T11:45:36+02:00
categories:
  - Machine Learning
tags:
  - Python
  - R
  - Clustering
---

---

K-means clustering is a method of vector quantization, originally from signal processing, that is popular for cluster analysis in data mining. k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells.

The problem is computationally difficult (NP-hard); however, there are efficient heuristic algorithms that are commonly employed and converge quickly to a local optimum. These are usually similar to the expectation-maximization algorithm for mixtures of Gaussian distributions via an iterative refinement approach employed by both algorithms. Additionally, they both use cluster centers to model the data; however, k-means clustering tends to find clusters of comparable spatial extent, while the expectation-maximization mechanism allows clusters to have different shapes.

# K-means clustering

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
# K-Means Clustering

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

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
~~~

In a clustering problem it always wise to determine how many clusters you should use for your model. We could put everyone in the same cluster or everyone in there own cluster. The optimal number of clusters is somewhere in between. Finding the optimal number of clusters can be done in several ways, for this problem we'll use the elbow method.

For the elbow method we'll compute the WCSS, Within Cluster Sum of Squares, for a possible number of clusters and use the elbow method o determine the optimal number of clusters.

I'm not gonna go in to much dept about this method, but for reference: [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)).

For this problem we assumed the optimal number of clusters was somewhere in between 1 and 10, and we assumed right! Genius. Plottin the WCSS gives:

![K-means clustering (Elbow Method)](/images/kmc1.png)

The elbow method gives us the optimal number of clusters, 5! After applying k-means to our dataset with 5 clusters, plotting the clusters, and giving the clusters the appropriate names, we get.

![K-means clustering](/images/kmc2.png)

## In R

~~~
# K-Means Clustering

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

# Using the elbow method to find the optimal number of clusters
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(dataset, i)$withinss)
plot(1:10,
     wcss,
     type = 'b',
     main = paste('The Elbow Method'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')

# Fitting K-Means to the dataset
set.seed(29)
kmeans = kmeans(x = dataset, centers = 5)
y_kmeans = kmeans$cluster

# Visualising the clusters
library(cluster)
clusplot(dataset,
         y_kmeans,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')
~~~

Produces somewhat worse pictures, but still correct.

![K-means clustering](/images/kmc3.png)

![K-means clustering](/images/kmc4.png)

So the mall should target the top right cluster, and maybe the careless ones aswell. Those are our ideal customers.

# K-means clustering notebook


# K-Means Clustering Example

Let's make some fake data that includes people clustered by income and age, randomly:


```python
from numpy import random, array

#Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range (k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 2.0)])
    X = array(X)
    return X
```

We'll use k-means to rediscover these clusters in unsupervised learning:


```python
%matplotlib inline

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random, float

data = createClusteredData(100, 5)

model = KMeans(n_clusters=5)

# Note I'm scaling the data to normalize it! Important for good results.
model = model.fit(scale(data))

# We can look at the clusters each data point was assigned to
print(model.labels_)

# And we'll visualize it:
plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))
plt.show()
```

    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3
     3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]



![png](/images/output_4_1.png)


## Activity

Things to play with: what happens if you don't scale the data? What happens if you choose different values of K? In the real world, you won't know the "right" value of K to start with - you'll need to converge on it yourself.
