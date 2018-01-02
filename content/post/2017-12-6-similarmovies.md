---
title: "Similar Movies"
date: 2017-12-06
categories:
  - Machine Learning
tags:
  - Python
  - R
  - Recommender systems
---

---


# Finding Similar Movies

We'll start by loading up the MovieLens dataset. Using Pandas, we can very quickly load the rows of the u.data and u.item files that we care about, and merge them together so we can work with movie names instead of ID's. (In a real production job, you'd stick with ID's and worry about the names at the display layer to make things more efficient. But this lets us understand what's going on better for now.)


```python
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('C:/Users/1asch/udemy-datascience/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('C:/Users/1asch/udemy-datascience/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)

```


```python
ratings.head()
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
      <th>movie_id</th>
      <th>title</th>
      <th>user_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>308</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>287</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>148</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>280</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>66</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Now the amazing pivot_table function on a DataFrame will construct a user / movie rating matrix. Note how NaN indicates missing data - movies that specific users didn't rate.


```python
movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
movieRatings.head()
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
      <th>title</th>
      <th>'Til There Was You (1997)</th>
      <th>1-900 (1994)</th>
      <th>101 Dalmatians (1996)</th>
      <th>12 Angry Men (1957)</th>
      <th>187 (1997)</th>
      <th>2 Days in the Valley (1996)</th>
      <th>20,000 Leagues Under the Sea (1954)</th>
      <th>2001: A Space Odyssey (1968)</th>
      <th>3 Ninjas: High Noon At Mega Mountain (1998)</th>
      <th>39 Steps, The (1935)</th>
      <th>...</th>
      <th>Yankee Zulu (1994)</th>
      <th>Year of the Horse (1997)</th>
      <th>You So Crazy (1994)</th>
      <th>Young Frankenstein (1974)</th>
      <th>Young Guns (1988)</th>
      <th>Young Guns II (1990)</th>
      <th>Young Poisoner's Handbook, The (1995)</th>
      <th>Zeus and Roxanne (1997)</th>
      <th>unknown</th>
      <th>Á köldum klaka (Cold Fever) (1994)</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1664 columns</p>
</div>



Let's extract a Series of users who rated Star Wars:


```python
starWarsRatings = movieRatings['Star Wars (1977)']
starWarsRatings.head()
```




    user_id
    0    5.0
    1    5.0
    2    5.0
    3    NaN
    4    5.0
    Name: Star Wars (1977), dtype: float64



Pandas' corrwith function makes it really easy to compute the pairwise correlation of Star Wars' vector of user rating with every other movie! After that, we'll drop any results that have no data, and construct a new DataFrame of movies and their correlation score (similarity) to Star Wars:


```python
similarMovies = movieRatings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
df.head(10)
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
      <th>0</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>0.872872</td>
    </tr>
    <tr>
      <th>1-900 (1994)</th>
      <td>-0.645497</td>
    </tr>
    <tr>
      <th>101 Dalmatians (1996)</th>
      <td>0.211132</td>
    </tr>
    <tr>
      <th>12 Angry Men (1957)</th>
      <td>0.184289</td>
    </tr>
    <tr>
      <th>187 (1997)</th>
      <td>0.027398</td>
    </tr>
    <tr>
      <th>2 Days in the Valley (1996)</th>
      <td>0.066654</td>
    </tr>
    <tr>
      <th>20,000 Leagues Under the Sea (1954)</th>
      <td>0.289768</td>
    </tr>
    <tr>
      <th>2001: A Space Odyssey (1968)</th>
      <td>0.230884</td>
    </tr>
    <tr>
      <th>39 Steps, The (1935)</th>
      <td>0.106453</td>
    </tr>
    <tr>
      <th>8 1/2 (1963)</th>
      <td>-0.142977</td>
    </tr>
  </tbody>
</table>
</div>



(That warning is safe to ignore.) Let's sort the results by similarity score, and we should have the movies most similar to Star Wars! Except... we don't. These results make no sense at all! This is why it's important to know your data - clearly we missed something important.


```python
similarMovies.sort_values(ascending=False)
```




    title
    Hollow Reed (1996)                                                                   1.000000
    Man of the Year (1995)                                                               1.000000
    Stripes (1981)                                                                       1.000000
    Full Speed (1996)                                                                    1.000000
    Golden Earrings (1947)                                                               1.000000
    Old Lady Who Walked in the Sea, The (Vieille qui marchait dans la mer, La) (1991)    1.000000
    Star Wars (1977)                                                                     1.000000
    Ed's Next Move (1996)                                                                1.000000
    Scarlet Letter, The (1926)                                                           1.000000
    Hurricane Streets (1998)                                                             1.000000
    Safe Passage (1994)                                                                  1.000000
    Outlaw, The (1943)                                                                   1.000000
    Twisted (1996)                                                                       1.000000
    Beans of Egypt, Maine, The (1994)                                                    1.000000
    Line King: Al Hirschfeld, The (1996)                                                 1.000000
    Mondo (1996)                                                                         1.000000
    Good Man in Africa, A (1994)                                                         1.000000
    No Escape (1994)                                                                     1.000000
    Cosi (1996)                                                                          1.000000
    Commandments (1997)                                                                  1.000000
    Last Time I Saw Paris, The (1954)                                                    1.000000
    Maya Lin: A Strong Clear Vision (1994)                                               1.000000
    Designated Mourner, The (1997)                                                       0.970725
    Albino Alligator (1996)                                                              0.968496
    Angel Baby (1995)                                                                    0.962250
    Prisoner of the Mountains (Kavkazsky Plennik) (1996)                                 0.927173
    Love in the Afternoon (1957)                                                         0.923381
    'Til There Was You (1997)                                                            0.872872
    A Chef in Love (1996)                                                                0.868599
    Quiet Room, The (1996)                                                               0.866025
                                                                                           ...   
    Pushing Hands (1992)                                                                -1.000000
    Lamerica (1994)                                                                     -1.000000
    Year of the Horse (1997)                                                            -1.000000
    Collectionneuse, La (1967)                                                          -1.000000
    Dream Man (1995)                                                                    -1.000000
    S.F.W. (1994)                                                                       -1.000000
    Nightwatch (1997)                                                                   -1.000000
    Squeeze (1996)                                                                      -1.000000
    Glass Shield, The (1994)                                                            -1.000000
    Slingshot, The (1993)                                                               -1.000000
    Lover's Knot (1996)                                                                 -1.000000
    Tough and Deadly (1995)                                                             -1.000000
    Sliding Doors (1998)                                                                -1.000000
    Show, The (1995)                                                                    -1.000000
    Nil By Mouth (1997)                                                                 -1.000000
    Fall (1997)                                                                         -1.000000
    Sudden Manhattan (1996)                                                             -1.000000
    Salut cousin! (1996)                                                                -1.000000
    Neon Bible, The (1995)                                                              -1.000000
    Crossfire (1947)                                                                    -1.000000
    Love and Death on Long Island (1997)                                                -1.000000
    For Ever Mozart (1996)                                                              -1.000000
    Swept from the Sea (1997)                                                           -1.000000
    Fille seule, La (A Single Girl) (1995)                                              -1.000000
    American Dream (1990)                                                               -1.000000
    Theodore Rex (1995)                                                                 -1.000000
    I Like It Like That (1994)                                                          -1.000000
    Two Deaths (1995)                                                                   -1.000000
    Roseanna's Grave (For Roseanna) (1997)                                              -1.000000
    Frankie Starlight (1995)                                                            -1.000000
    Length: 1410, dtype: float64



Our results are probably getting messed up by movies that have only been viewed by a handful of people who also happened to like Star Wars. So we need to get rid of movies that were only watched by a few people that are producing spurious results. Let's construct a new DataFrame that counts up how many ratings exist for each movie, and also the average rating while we're at it - that could also come in handy later.


```python
import numpy as np
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
movieStats.head()
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
    <tr>
      <th></th>
      <th colspan="2" halign="left">rating</th>
    </tr>
    <tr>
      <th></th>
      <th>size</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>9</td>
      <td>2.333333</td>
    </tr>
    <tr>
      <th>1-900 (1994)</th>
      <td>5</td>
      <td>2.600000</td>
    </tr>
    <tr>
      <th>101 Dalmatians (1996)</th>
      <td>109</td>
      <td>2.908257</td>
    </tr>
    <tr>
      <th>12 Angry Men (1957)</th>
      <td>125</td>
      <td>4.344000</td>
    </tr>
    <tr>
      <th>187 (1997)</th>
      <td>41</td>
      <td>3.024390</td>
    </tr>
  </tbody>
</table>
</div>



Let's get rid of any movies rated by fewer than 100 people, and check the top-rated ones that are left:


```python
popularMovies = movieStats['rating']['size'] >= 100
movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15]
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
    <tr>
      <th></th>
      <th colspan="2" halign="left">rating</th>
    </tr>
    <tr>
      <th></th>
      <th>size</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Close Shave, A (1995)</th>
      <td>112</td>
      <td>4.491071</td>
    </tr>
    <tr>
      <th>Schindler's List (1993)</th>
      <td>298</td>
      <td>4.466443</td>
    </tr>
    <tr>
      <th>Wrong Trousers, The (1993)</th>
      <td>118</td>
      <td>4.466102</td>
    </tr>
    <tr>
      <th>Casablanca (1942)</th>
      <td>243</td>
      <td>4.456790</td>
    </tr>
    <tr>
      <th>Shawshank Redemption, The (1994)</th>
      <td>283</td>
      <td>4.445230</td>
    </tr>
    <tr>
      <th>Rear Window (1954)</th>
      <td>209</td>
      <td>4.387560</td>
    </tr>
    <tr>
      <th>Usual Suspects, The (1995)</th>
      <td>267</td>
      <td>4.385768</td>
    </tr>
    <tr>
      <th>Star Wars (1977)</th>
      <td>584</td>
      <td>4.359589</td>
    </tr>
    <tr>
      <th>12 Angry Men (1957)</th>
      <td>125</td>
      <td>4.344000</td>
    </tr>
    <tr>
      <th>Citizen Kane (1941)</th>
      <td>198</td>
      <td>4.292929</td>
    </tr>
    <tr>
      <th>To Kill a Mockingbird (1962)</th>
      <td>219</td>
      <td>4.292237</td>
    </tr>
    <tr>
      <th>One Flew Over the Cuckoo's Nest (1975)</th>
      <td>264</td>
      <td>4.291667</td>
    </tr>
    <tr>
      <th>Silence of the Lambs, The (1991)</th>
      <td>390</td>
      <td>4.289744</td>
    </tr>
    <tr>
      <th>North by Northwest (1959)</th>
      <td>179</td>
      <td>4.284916</td>
    </tr>
    <tr>
      <th>Godfather, The (1972)</th>
      <td>413</td>
      <td>4.283293</td>
    </tr>
  </tbody>
</table>
</div>



100 might still be too low, but these results look pretty good as far as "well rated movies that people have heard of." Let's join this data with our original set of similar movies to Star Wars:


```python
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
```

    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\reshape\merge.py:551: UserWarning: merging between different levels can give an unintended result (2 levels on the left, 1 on the right)
      warnings.warn(msg, UserWarning)



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
      <th>(rating, size)</th>
      <th>(rating, mean)</th>
      <th>similarity</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101 Dalmatians (1996)</th>
      <td>109</td>
      <td>2.908257</td>
      <td>0.211132</td>
    </tr>
    <tr>
      <th>12 Angry Men (1957)</th>
      <td>125</td>
      <td>4.344000</td>
      <td>0.184289</td>
    </tr>
    <tr>
      <th>2001: A Space Odyssey (1968)</th>
      <td>259</td>
      <td>3.969112</td>
      <td>0.230884</td>
    </tr>
    <tr>
      <th>Absolute Power (1997)</th>
      <td>127</td>
      <td>3.370079</td>
      <td>0.085440</td>
    </tr>
    <tr>
      <th>Abyss, The (1989)</th>
      <td>151</td>
      <td>3.589404</td>
      <td>0.203709</td>
    </tr>
  </tbody>
</table>
</div>



And, sort these new results by similarity score. That's more like it!


```python
df.sort_values(['similarity'], ascending=False)[:15]
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
      <th>(rating, size)</th>
      <th>(rating, mean)</th>
      <th>similarity</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Star Wars (1977)</th>
      <td>584</td>
      <td>4.359589</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Empire Strikes Back, The (1980)</th>
      <td>368</td>
      <td>4.206522</td>
      <td>0.748353</td>
    </tr>
    <tr>
      <th>Return of the Jedi (1983)</th>
      <td>507</td>
      <td>4.007890</td>
      <td>0.672556</td>
    </tr>
    <tr>
      <th>Raiders of the Lost Ark (1981)</th>
      <td>420</td>
      <td>4.252381</td>
      <td>0.536117</td>
    </tr>
    <tr>
      <th>Austin Powers: International Man of Mystery (1997)</th>
      <td>130</td>
      <td>3.246154</td>
      <td>0.377433</td>
    </tr>
    <tr>
      <th>Sting, The (1973)</th>
      <td>241</td>
      <td>4.058091</td>
      <td>0.367538</td>
    </tr>
    <tr>
      <th>Indiana Jones and the Last Crusade (1989)</th>
      <td>331</td>
      <td>3.930514</td>
      <td>0.350107</td>
    </tr>
    <tr>
      <th>Pinocchio (1940)</th>
      <td>101</td>
      <td>3.673267</td>
      <td>0.347868</td>
    </tr>
    <tr>
      <th>Frighteners, The (1996)</th>
      <td>115</td>
      <td>3.234783</td>
      <td>0.332729</td>
    </tr>
    <tr>
      <th>L.A. Confidential (1997)</th>
      <td>297</td>
      <td>4.161616</td>
      <td>0.319065</td>
    </tr>
    <tr>
      <th>Wag the Dog (1997)</th>
      <td>137</td>
      <td>3.510949</td>
      <td>0.318645</td>
    </tr>
    <tr>
      <th>Dumbo (1941)</th>
      <td>123</td>
      <td>3.495935</td>
      <td>0.317656</td>
    </tr>
    <tr>
      <th>Bridge on the River Kwai, The (1957)</th>
      <td>165</td>
      <td>4.175758</td>
      <td>0.316580</td>
    </tr>
    <tr>
      <th>Philadelphia Story, The (1940)</th>
      <td>104</td>
      <td>4.115385</td>
      <td>0.314272</td>
    </tr>
    <tr>
      <th>Miracle on 34th Street (1994)</th>
      <td>101</td>
      <td>3.722772</td>
      <td>0.310921</td>
    </tr>
  </tbody>
</table>
</div>



Ideally we'd also filter out the movie we started from - of course Star Wars is 100% similar to itself. But otherwise these results aren't bad.

## Activity

100 was an arbitrarily chosen cutoff. Try different values - what effect does it have on the end results?


```python

```
