---
title: "Reinforcement Learning"
date: 2017-09-24T11:45:36+02:00
categories:
  - Machine Learning
tags:
  - Python
  - R
  - Reinforcement Learning
---

---

Reinforcement learning (RL) is an area of machine learning inspired by behaviorist psychology, concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward. The problem, due to its generality, is studied in many other disciplines, such as game theory, control theory, operations research, information theory, simulation-based optimization, multi-agent systems, swarm intelligence, statistics and genetic algorithms. In the operations research and control literature, the field where reinforcement learning methods are studied is called approximate dynamic programming. The problem has been studied in the theory of optimal control, though most studies are concerned with the existence of optimal solutions and their characterization, and not with the learning or approximation aspects. In economics and game theory, reinforcement learning may be used to explain how equilibrium may arise under bounded rationality.


# Reinforcement Learning

In this problem we are hired by the car company who wants to do social network ads. We've selected the right people for the car company, but the marketing team came up with 10 ads to show them. How do we know what ad to show? The one on the bridge, the mountain, or the one with the hot babe?

The easiest way to learn this is to just A/B test this out, but we don't have money to do the campaign twice. So we'll have to find the best ad on the job. To show that this is very, very, very usefull we'll show you the example of random advertising first.

~~~
# Random Selection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
~~~

total_reward = 1217. So 1217 people click on the adverstisement. Not to shabby, but I bet we can do much better.

![Random Selection](/images/rl1.png)

The graph shows how many times each ad has been chosen. If we chose wisely we can improve this for certain.

## Multi-armed bandit problem

In this section we will also find a way to solve the [Multi-armed bandit problem](https://en.wikipedia.org/wiki/Multi-armed_bandit). Here's a picture of a one armed bandit:

![One armed bandit](/images/banditslot1.jpg)

The multi-armed bandit problem is widely known problem by gamblers. It's called the armed bandit, because it has one arm and it robs you blind.

Now imagine we play on multiple machines which all have different win-rate distributions. They're all pour, but if we play wisely we might be able to win or at least lose the least amount possible.

Million dollar question is: Which machine has the best pay-out distribution?

## The dataset Ads_CTR_Optimisation

[Market_Basket_Optimisation](/data/Ads_CTR_Optimisation.csv)

Since it is a dataset containing 10000 entry's we only show the top and bottom part of the Ads_CTR_Optimisation.

| Ad 1 | Ad 2 | Ad 3 | Ad 4 | Ad 5 | Ad 6 | Ad 7 | Ad 8 | Ad 9 | Ad 10 |
|------|------|------|------|------|------|------|------|------|-------|
| 1    | 0    | 0    | 0    | 1    | 0    | 0    | 0    | 1    | 0     |
| 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1    | 0     |
| 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| 0    | 1    | 0    | 0    | 0    | 0    | 0    | 1    | 0    | 0     |
| 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| 1    | 1    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| 0    | 0    | 0    | 1    | 0    | 0    | 0    | 0    | 0    | 0     |
| 1    | 1    | 0    | 0    | 1    | 0    | 0    | 0    | 0    | 0     |
| 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| 0    | 0    | 1    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
|......|......|......|......|......|......|......|......|......|.......|
| 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| 0    | 0    | 0    | 1    | 0    | 0    | 0    | 0    | 0    | 0     |
| 0    | 1    | 0    | 1    | 1    | 0    | 1    | 0    | 0    | 0     |
| 0    | 0    | 0    | 1    | 0    | 0    | 1    | 0    | 0    | 0     |
| 0    | 0    | 0    | 0    | 1    | 0    | 0    | 0    | 1    | 0     |
| 0    | 0    | 1    | 0    | 0    | 0    | 0    | 0    | 1    | 0     |
| 0    | 0    | 1    | 0    | 0    | 0    | 0    | 1    | 0    | 0     |
| 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| 1    | 0    | 0    | 0    | 0    | 0    | 0    | 1    | 0    | 0     |
| 0    | 1    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |

Where 1 means the ad has been clicked and 0 ignored. Notice this marketing problem is the same as the multi-armed bandit problem with 10 machines. Now we have to figure out which ad/machine has the highest click/win rate!

## In Python

To make a better selection than random we'll use the upper confidence bound algoritm. Given below.

![upper confidence bound](/images/rl2.png)

Notice during every round we select the ad that is 'most succesfull' or has the highest expected return value. Machines or ads that don't do well get left behind quite easily by this algoritm, while machines or ads that do well stand out and get selected!

~~~
# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

~~~

Gives total_reward = 2178! So we almost doubled our clicks. The graph below shows the times each ad was picked by our algorithm. So we can clearly see that ad 5 had the highest upper bound lots of team in the rounds. So it's most likely the most succesfull ad.

![Hist of ads selections](/images/rl3.png)

So the best ad is 5!

## In R

Random selection:

~~~
# Random Selection

# Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
N = 10000
d = 10
ads_selected = integer(0)
total_reward = 0
for (n in 1:N) {
  ad = sample(1:10, 1)
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  total_reward = total_reward + reward
}

# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')
~~~

To make a better selection than random we'll use the upper confidence bound algoritm.

~~~
# Upper Confidence Bound

# Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
N = 10000
d = 10
ads_selected = integer(0)
numbers_of_selections = integer(d)
sums_of_rewards = integer(d)
total_reward = 0
for (n in 1:N) {
  ad = 0
  max_upper_bound = 0
  for (i in 1:d) {
    if (numbers_of_selections[i] > 0) {
      average_reward = sums_of_rewards[i] / numbers_of_selections[i]
      delta_i = sqrt(3/2 * log(n) / numbers_of_selections[i])
      upper_bound = average_reward + delta_i
    } else {
      upper_bound = 1e400
    }
    if (upper_bound > max_upper_bound) {
      max_upper_bound = upper_bound
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  reward = dataset[n, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')
~~~

![Histogram of ads selections](/images/rl4.png)

With an increase of 800 selected ads.

# Thompson Sampling algorithm

In artificial intelligence, Thompson sampling,[1] named after William R. Thompson, is a heuristic for choosing actions that addresses the exploration-exploitation dilemma in the multi-armed bandit problem. It consists in choosing the action that maximizes the expected reward with respect to a randomly drawn belief

![Thompson Sampling algorithm](/images/rl5.png)

Where instead of taking the ad with the highest upper bound, we take the add with the highest expected beta distribution.

## Bayesian Inference

Now here it gets a bit mathematical, but if you just think about it intuitively it's quite logical.

![Thompson Sampling algorithm](/images/rl6.png)

With random distributions the algorithm converges much quicker to the best machine or ad. Bad ads get really low expected click rates very quickly.

## In Python
~~~
# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

# Visualising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
~~~

Gives a total_reward of 2610! Number of ads selections

![Histogram of ads selections](/images/rl7.png)

## In R

~~~
# Thompson Sampling

# Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
N = 10000
d = 10
ads_selected = integer(0)
numbers_of_rewards_1 = integer(d)
numbers_of_rewards_0 = integer(d)
total_reward = 0
for (n in 1:N) {
  ad = 0
  max_random = 0
  for (i in 1:d) {
    random_beta = rbeta(n = 1,
                        shape1 = numbers_of_rewards_1[i] + 1,
                        shape2 = numbers_of_rewards_0[i] + 1)
    if (random_beta > max_random) {
      max_random = random_beta
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  if (reward == 1) {
    numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
  } else {
    numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
  }
  total_reward = total_reward + reward
}

# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')
~~~

![Histogram of ads selections](/images/rl8.png)

with 2589 succesfull clicks! So by selecting our ad in a smart way we managed to increase our clicks by a whopping 1500! That's some easy money made, or saved!

# Q learning essentials

So in the above example we've figured out a way to find the best add as soon as possible, but reinforcement learning is also used in a lot of other problems. One of my favourite usage of RL is the one where we have a hero or agent! This agent is dealing with a set of environmental states $s$. A set of possible actions in those states $a$. And a value of each state/action $Q$. Start off with $Q$ values of 0, explore the space, as bad things happen after a give state/action, reduce its $Q$. As rewards happen after a given state/action, increase its $Q$. This is called **Q Learning** and is used in AlphaZero, AI Pacman, AI Tetris and lots of other games! Exciting stuff.

$$Q(s,a) += \alpha\cdot(reward(s,a) + max(Q(s') - Q(s,a)))$$

where $s$ is the previous state $a$ is the previous action, $s'$ is the current state, and $\alpha$ is the discount factor.

Superlinks:
https://inst.eecs.berkeley.edu/~cs188/sp12/projects/reinforcement/reinforcement.html
https://github.com/studywolf/blog/tree/master/RL/Cat%20vs%20Mouse%20exploration
http://pymdptoolbox.readthedocs.io/en/latest/api/mdp.html
