---
title: "Association Rule Learning"
date: 2017-09-24T11:45:36+02:00
categories:
  - Machine Learning
tags:
  - Python
  - R
  - Recommender systems
---

---

Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large databases. It is intended to identify strong rules discovered in databases using some measures of interestingness.[1] Based on the concept of strong rules, Rakesh Agrawal, Tomasz Imieliński and Arun Swami [2] introduced association rules for discovering regularities between products in large-scale transaction data recorded by point-of-sale (POS) systems in supermarkets. For example, the rule {\displaystyle \{\mathrm {onions,potatoes} \}\Rightarrow \{\mathrm {burger} \}} \{{\mathrm  {onions,potatoes}}\}\Rightarrow \{{\mathrm  {burger}}\} found in the sales data of a supermarket would indicate that if a customer buys onions and potatoes together, they are likely to also buy hamburger meat. Such information can be used as the basis for decisions about marketing activities such as, e.g., promotional pricing or product placements. In addition to the above example from market basket analysis association rules are employed today in many application areas including Web usage mining, intrusion detection, continuous production, and bioinformatics. In contrast with sequence mining, association rule learning typically does not consider the order of items either within a transaction or across transactions.

# Association Rule Learning

We're a data scientist for a supermarket and we're asked to analyse what product are bought together frequently. This way we are able to place our products wisely so they get bought together more often.

If the supermarket is kind of a dick and knows which products are always bought together, it could also place the products at different sides of the store, so customers will have to walk all the way through the store, resulting to buy more products. If you work at such a store... resign.

Association rule learning has taught us many beautifull things. Like beer and diapers are frequently bought together during the weekends! Why? Because whenever it's weekend and a couple runs out of diapers, who has to get them? That's right, the man of the house, and he brings some? That's right beer!

## The dataset Market_Basket_Optimisation

[Market_Basket_Optimisation](/data/Market_Basket_Optimisation.csv)

Since it is a dataset containing 7501 entry's we only show the top and bottom part of the Market_Basket_Optimisation.

|                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |
|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| shrimp               | almonds              | avocado              | vegetables mix       | green grapes         | whole weat flour     | yams                 | cottage cheese       | energy drink         | tomato juice         | low fat yogurt       | green tea            |
| burgers              | meatballs            | eggs                 |                      |                      |                      |                      |                      |                      |                      |                      |                      |
| chutney              |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |
| turkey               | avocado              |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |
| mineral water        | milk                 | energy bar           | whole wheat rice     | green tea            |                      |                      |                      |                      |                      |                      |                      |  
| low fat yogurt       |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                 
| whole wheat pasta    | french fries         |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                  
| soup                 | light cream          | shallot              |                      |                      |                      |                      |                      |                      |                      |                      |                      |              
| frozen vegetables    | spaghetti            | green tea            |                      |                      |                      |                      |                      |                      |                      |                      |                      |                 
| french fries         |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |               
|	...	       |	...	      |		...	     |		...	    |		...        |		...	  | 		...	 |	...		|	...	       |	...	      |		...	     |		...	    |
| herb & pepper        |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                 
| chocolate            | escalope             |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                  
| burgers              | salmon               | pancakes             | french fries         | frozen smoothie      | fresh bread          | mint                 |                      |                      |                      |                      |                      |                  
| turkey               | burgers              | dessert wine         | shrimp               | pasta                | tomatoes             | pepper               | milk                 | pancakes             | whole wheat rice     | oil                  | frozen smoothie      |                   
| pancakes             | light mayo           |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                  
| butter               | light mayo           | fresh bread          |                      |                      |                      |                      |                      |                      |                      |                      |                      |                  
| burgers              | frozen vegetables    | eggs                 | french fries         | magazines            | green tea            |                      |                      |                      |                      |                      |                      |                
| chicken              |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                  
| escalope             | green tea            |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                   
| eggs                 | frozen smoothie      | yogurt cake          | low fat yogurt       |                      |                      |                      |                      |                      |                      |                      |                      |


## In R

~~~
# Apriori

# Data Preprocessing
#install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])
~~~

Summary(dataset) gives:

~~~
transactions as itemMatrix in sparse format with
 7501 rows (elements/itemsets/transactions) and
 119 columns (items) and a density of 0.03288973

most frequent items:
mineral water          eggs     spaghetti  french fries     chocolate
         1788          1348          1306          1282          1229
      (Other)
        22405

element (itemset/transaction) length distribution:
sizes
   1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
1754 1358 1044  816  667  493  391  324  259  139  102   67   40   22   17
  16   18   19   20
   4    1    2    1

   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
  1.000   2.000   3.000   3.914   5.000  20.000

includes extended item information - examples:
             labels
1           almonds
2 antioxydant juice
3         asparagus
~~~

The supermarket asked us to only look at the products which are bought frequently, for example 3 times a day. Since this is a dataset of all the transactions of an entire week we a have minimum support value of 3*7/7500=0.003. The confidence is the percentage that this association is true. So in our case 0.20 gives a lot of results. Setting the confidence to 0.8 for example will give no results.

Looking at the highest lift gives us the best association rules.

![apriori - lift](/images/arl1.png)

~~~
> inspect(sort(rules, by = 'lift')[1:10])
     lhs                       rhs                 support confidence     lift count
[1]  {light cream}          => {chicken}       0.004532729  0.2905983 4.843951    34
[2]  {pasta}                => {escalope}      0.005865885  0.3728814 4.700812    44
[3]  {pasta}                => {shrimp}        0.005065991  0.3220339 4.506672    38
[4]  {eggs,                                                                         
      ground beef}          => {herb & pepper} 0.004132782  0.2066667 4.178455    31
[5]  {whole wheat pasta}    => {olive oil}     0.007998933  0.2714932 4.122410    60
[6]  {herb & pepper,                                                                
      spaghetti}            => {ground beef}   0.006399147  0.3934426 4.004360    48
[7]  {herb & pepper,                                                                
      mineral water}        => {ground beef}   0.006665778  0.3906250 3.975683    50
[8]  {tomato sauce}         => {ground beef}   0.005332622  0.3773585 3.840659    40
[9]  {mushroom cream sauce} => {escalope}      0.005732569  0.3006993 3.790833    43
[10] {frozen vegetables,                                                            
      mineral water,                                                                
      spaghetti}            => {ground beef}   0.004399413  0.3666667 3.731841    33
~~~

## In Python

~~~
# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
~~~

In which we first transform the dataset in the right format, a list. After we use apyori a python function we run locally and can be downloaded here.

[apyori.py](/data/apyori.py)

Gives the same results as in R.

# Eclat

Eclat[9] (alt. ECLAT, stands for Equivalence Class Transformation) is a depth-first search algorithm using set intersection. It is a naturally elegant algorithm suitable for both sequential as well as parallel execution with locality-enhancing properties. It was first introduced by Zaki, Parthasarathy, Li and Ogihara in a series of papers written in 1997.

Eclat only uses support, not lift.

## Support
![Eclat - Support](/images/arl2.png)

## In R

~~~
# Eclat

# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv')
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat on the dataset
rules = eclat(data = dataset, parameter = list(support = 0.004, minlen = 2))

# Visualising the results
inspect(sort(rules, by = 'support')[1:10])
~~~

gives

~~~
> inspect(sort(rules, by = 'support')[1:10])
     items                             support    count
[1]  {mineral water,spaghetti}         0.05972537 448  
[2]  {chocolate,mineral water}         0.05265965 395  
[3]  {eggs,mineral water}              0.05092654 382  
[4]  {milk,mineral water}              0.04799360 360  
[5]  {ground beef,mineral water}       0.04092788 307  
[6]  {ground beef,spaghetti}           0.03919477 294  
[7]  {chocolate,spaghetti}             0.03919477 294  
[8]  {eggs,spaghetti}                  0.03652846 274  
[9]  {eggs,french fries}               0.03639515 273  
[10] {frozen vegetables,mineral water} 0.03572857 268  
~~~
