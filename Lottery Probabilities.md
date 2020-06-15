
# Logic For Mobile App For Lottery Addiction

I wish to create the underlying logic for an app that helps treat those addicted to played the lottery by showing them the odds. For this version of the app, I will focus on the [6/49 lottery](https://en.wikipedia.org/wiki/Lotto_6/49).

I will also consider [historical data](https://www.kaggle.com/datascienceai/lottery-dataset) coming from the national 6/49 lottery game in Canada. 

## Core Functions

I will write two functions that will be used frequently: combination and factorial calculators. 


```python
def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

def combinations(n, k):
    numerator = factorial(n)
    denominator = factorial(k) * factorial(n-k)
    return numerator / denominator
```

## One Ticket Probability

Below I will build a function that calculates the probability of winning the big prize for any given ticket. For each drawing, six numbers are drawn from a set of 49, and a player wins the big prize if the six numbers on their tickets match all six numbers.


```python
def one_ticket_probability(user_numbers): 
    n_outcomes = combinations(49, 6)
    probability_one_ticket = 1/n_outcomes
    percentage_form = probability_one_ticket * 100
    
    print('''Your chances to win the big prize with the numbers {} are {:.7f}%.
In other words, you have a 1 in {:,} chances to win.'''.format(user_numbers,
                    percentage_form, int(n_outcomes)))
```

I will test the function with a few inputs. 


```python
input1 = [1, 2, 3, 4, 5, 6]
one_ticket_probability(input1)
```

    Your chances to win the big prize with the numbers [1, 2, 3, 4, 5, 6] are 0.0000072%.
    In other words, you have a 1 in 13,983,816 chances to win.


## Historical Data 

I also want users to be able to compare their ticket against the historical lottery data in Canada and determine whether they would have ever won by now. First I will view the data


```python
import pandas as pd
sixfournine = pd.read_csv("649.csv")
print(sixfournine.shape)
```

    (3665, 11)



```python
sixfournine.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PRODUCT</th>
      <th>DRAW NUMBER</th>
      <th>SEQUENCE NUMBER</th>
      <th>DRAW DATE</th>
      <th>NUMBER DRAWN 1</th>
      <th>NUMBER DRAWN 2</th>
      <th>NUMBER DRAWN 3</th>
      <th>NUMBER DRAWN 4</th>
      <th>NUMBER DRAWN 5</th>
      <th>NUMBER DRAWN 6</th>
      <th>BONUS NUMBER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>649</td>
      <td>1</td>
      <td>0</td>
      <td>6/12/1982</td>
      <td>3</td>
      <td>11</td>
      <td>12</td>
      <td>14</td>
      <td>41</td>
      <td>43</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>649</td>
      <td>2</td>
      <td>0</td>
      <td>6/19/1982</td>
      <td>8</td>
      <td>33</td>
      <td>36</td>
      <td>37</td>
      <td>39</td>
      <td>41</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>649</td>
      <td>3</td>
      <td>0</td>
      <td>6/26/1982</td>
      <td>1</td>
      <td>6</td>
      <td>23</td>
      <td>24</td>
      <td>27</td>
      <td>39</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>649</td>
      <td>4</td>
      <td>0</td>
      <td>7/3/1982</td>
      <td>3</td>
      <td>9</td>
      <td>10</td>
      <td>13</td>
      <td>20</td>
      <td>43</td>
      <td>34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>649</td>
      <td>5</td>
      <td>0</td>
      <td>7/10/1982</td>
      <td>5</td>
      <td>14</td>
      <td>21</td>
      <td>31</td>
      <td>34</td>
      <td>47</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>




```python
sixfournine.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PRODUCT</th>
      <th>DRAW NUMBER</th>
      <th>SEQUENCE NUMBER</th>
      <th>DRAW DATE</th>
      <th>NUMBER DRAWN 1</th>
      <th>NUMBER DRAWN 2</th>
      <th>NUMBER DRAWN 3</th>
      <th>NUMBER DRAWN 4</th>
      <th>NUMBER DRAWN 5</th>
      <th>NUMBER DRAWN 6</th>
      <th>BONUS NUMBER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3660</th>
      <td>649</td>
      <td>3587</td>
      <td>0</td>
      <td>6/6/2018</td>
      <td>10</td>
      <td>15</td>
      <td>23</td>
      <td>38</td>
      <td>40</td>
      <td>41</td>
      <td>35</td>
    </tr>
    <tr>
      <th>3661</th>
      <td>649</td>
      <td>3588</td>
      <td>0</td>
      <td>6/9/2018</td>
      <td>19</td>
      <td>25</td>
      <td>31</td>
      <td>36</td>
      <td>46</td>
      <td>47</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3662</th>
      <td>649</td>
      <td>3589</td>
      <td>0</td>
      <td>6/13/2018</td>
      <td>6</td>
      <td>22</td>
      <td>24</td>
      <td>31</td>
      <td>32</td>
      <td>34</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3663</th>
      <td>649</td>
      <td>3590</td>
      <td>0</td>
      <td>6/16/2018</td>
      <td>2</td>
      <td>15</td>
      <td>21</td>
      <td>31</td>
      <td>38</td>
      <td>49</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3664</th>
      <td>649</td>
      <td>3591</td>
      <td>0</td>
      <td>6/20/2018</td>
      <td>14</td>
      <td>24</td>
      <td>31</td>
      <td>35</td>
      <td>37</td>
      <td>48</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>



## Function for Historical Data Check

I will now build the historical data check function described above. 


```python
def extract_number(row):
    row = row[4:10]
    row = set(row.values)
    return row

winning_numbers = sixfournine.apply(extract_number, axis=1)
winning_numbers.head()
```




    0    {3, 41, 11, 12, 43, 14}
    1    {33, 36, 37, 39, 8, 41}
    2     {1, 6, 39, 23, 24, 27}
    3     {3, 9, 10, 43, 13, 20}
    4    {34, 5, 14, 47, 21, 31}
    dtype: object




```python
def check_historical_occurence(user_nums, winning_nums):
    user_nums_set = set(user_nums)
    bools = winning_nums == user_nums_set
    total = bools.sum()
    if total == 0:
        print('''The combination {} has never occured.
This doesn't mean it's more likely to occur now. Your chances to win the big prize in the next drawing using the combination {} are 0.0000072%.
In other words, you have a 1 in 13,983,816 chances to win.'''.format(user_nums, user_nums))
        
    else:
        print('''The number of times combination {} has occured in the past is {}.
Your chances to win the big prize in the next drawing using the combination {} are 0.0000072%.
In other words, you have a 1 in 13,983,816 chances to win.'''.format(user_nums, total,
                                                                            user_nums))
        
    
    
```

I will now test the function


```python
user_numbs = [1, 2, 3, 4, 5, 6]
check_historical_occurence(user_numbs, winning_numbers)
```

    The combination [1, 2, 3, 4, 5, 6] has never occured.
    This doesn't mean it's more likely to occur now. Your chances to win the big prize in the next drawing using the combination [1, 2, 3, 4, 5, 6] are 0.0000072%.
    In other words, you have a 1 in 13,983,816 chances to win.



```python
user_numbs2 = [33, 36, 37, 39, 8, 41]
check_historical_occurence(user_numbs2, winning_numbers)
```

    The number of times combination [33, 36, 37, 39, 8, 41] has occured in the past is 1.
    Your chances to win the big prize in the next drawing using the combination [33, 36, 37, 39, 8, 41] are 0.0000072%.
    In other words, you have a 1 in 13,983,816 chances to win.


## Multi Ticket Probability

I also want users to put in multiple tickets and view the probability of winning. 

The multi_ticket_probability() function below takes in the number of tickets and prints probability information depending on the input.


```python
def multi_ticket_probability(n_tickets):
    #total number of outcomes
    outcomes = combinations(49, 6)
    prob = n_tickets / outcomes
    prob_percent = prob * 100
    if n_tickets == 1:
        print('''Your chances to win the big prize with one ticket are {:.6f}%.
In other words, you have a 1 in {:,} chances to win.'''.format(prob_percent, int(outcomes)))
    
    else:
        combinations_simplified = round(outcomes / n_tickets)   
        print('''Your chances to win the big prize with {:,} different tickets are {:.6f}%.
In other words, you have a 1 in {:,} chances to win.'''.format(n_tickets, prob_percent,
                                                               combinations_simplified))
```

I will now test my function


```python
multi_ticket_probability(1)
```

    Your chances to win the big prize with one ticket are 0.000007%.
    In other words, you have a 1 in 13,983,816 chances to win.



```python
multi_ticket_probability(100)
```

    Your chances to win the big prize with 100 different tickets are 0.000715%.
    In other words, you have a 1 in 139,838 chances to win.



```python

```
