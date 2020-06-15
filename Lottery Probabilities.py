#!/usr/bin/env python
# coding: utf-8

# # Logic For Mobile App For Lottery Addiction
# 
# I wish to create the underlying logic for an app that helps treat those addicted to played the lottery by showing them the odds. For this version of the app, I will focus on the [6/49 lottery](https://en.wikipedia.org/wiki/Lotto_6/49).
# 
# I will also consider [historical data](https://www.kaggle.com/datascienceai/lottery-dataset) coming from the national 6/49 lottery game in Canada. 

# ## Core Functions
# 
# I will write two functions that will be used frequently: combination and factorial calculators. 

# In[5]:


def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

def combinations(n, k):
    numerator = factorial(n)
    denominator = factorial(k) * factorial(n-k)
    return numerator / denominator


# ## One Ticket Probability
# 
# Below I will build a function that calculates the probability of winning the big prize for any given ticket. For each drawing, six numbers are drawn from a set of 49, and a player wins the big prize if the six numbers on their tickets match all six numbers.

# In[9]:


def one_ticket_probability(user_numbers): 
    n_outcomes = combinations(49, 6)
    probability_one_ticket = 1/n_outcomes
    percentage_form = probability_one_ticket * 100
    
    print('''Your chances to win the big prize with the numbers {} are {:.7f}%.
In other words, you have a 1 in {:,} chances to win.'''.format(user_numbers,
                    percentage_form, int(n_outcomes)))


# I will test the function with a few inputs. 

# In[10]:


input1 = [1, 2, 3, 4, 5, 6]
one_ticket_probability(input1)


# ## Historical Data 
# 
# I also want users to be able to compare their ticket against the historical lottery data in Canada and determine whether they would have ever won by now. First I will view the data

# In[11]:


import pandas as pd
sixfournine = pd.read_csv("649.csv")
print(sixfournine.shape)


# In[13]:


sixfournine.head()


# In[14]:


sixfournine.tail()


# ## Function for Historical Data Check
# 
# I will now build the historical data check function described above. 

# In[17]:


def extract_number(row):
    row = row[4:10]
    row = set(row.values)
    return row

winning_numbers = sixfournine.apply(extract_number, axis=1)
winning_numbers.head()


# In[27]:


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
        
    
    


# I will now test the function

# In[28]:


user_numbs = [1, 2, 3, 4, 5, 6]
check_historical_occurence(user_numbs, winning_numbers)


# In[29]:


user_numbs2 = [33, 36, 37, 39, 8, 41]
check_historical_occurence(user_numbs2, winning_numbers)


# ## Multi Ticket Probability
# 
# I also want users to put in multiple tickets and view the probability of winning. 
# 
# The multi_ticket_probability() function below takes in the number of tickets and prints probability information depending on the input.

# In[30]:


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


# I will now test my function

# In[31]:


multi_ticket_probability(1)


# In[32]:


multi_ticket_probability(100)


# In[ ]:




