#!/usr/bin/env python
# coding: utf-8

# # Predicting Bike Rentals Using Random Forests
# 
# In this project, I will predict the total number of bike rentals in a given hour. I will use the [bike sharing data from UC Irvine](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) that includes information regarding bike rentals from Washington D.C.. 

# # Exploring the Data

# In[1]:


import pandas as pd

bike_rentals = pd.read_csv("bike_rental_hour.csv")
print(bike_rentals.shape)
bike_rentals.head()


# cnt describes the total number of bikes rented, and this is what I will want to predict. Let's first visualize this data

# In[2]:


get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

plt.hist(bike_rentals["cnt"])


# I will now explore how each column is correlated with cnt

# In[4]:


bike_rentals.corr()["cnt"]


# It looks like registered and casual are highly correlated with cnt, which makes sense since cnt is the sum of registered and casual. 

# ## Introducing Features
# 
# Before applying machine learning models, I will calculate features. In particular, I will introduce some order in the the time of day by creating a new column with labels for morning, afternoon, evening, and night. This will bundle similar times together, enabling the model to make better predictions. 

# In[6]:


def assign_label(hour):
    if hour >=0 and hour < 6:
        return 4 #night
    elif hour >=6 and hour < 12:
        return 1 #morning
    elif hour >= 12 and hour < 18:
        return 2 #afternoon
    elif hour >= 18 and hour <=24:
        return 3 #evening
    
bike_rentals["time_label"] = bike_rentals["hr"].apply(assign_label)

bike_rentals.head()


# ## Splitting the Data into Training and Testing Sets
# 
# I will split the data into a training and testing set using a 80/20 split. I will use the mean square error metric to evaluate my error since our data is continuous. 

# In[ ]:


train = bike_rentals.sample(frac=.8)
test = bike_rentals.loc[~bike_rentals.index.isin(train.index)]


# ## Applying Linear Regression to the Data
# 
# Again, I will ignore the sacual and registered columns because cnt is derived from them. 

# In[32]:


train_cols = list(train.columns.drop(['cnt', 'casual', 'dteday', 'registered']))
train_cols


# In[35]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg = LinearRegression().fit(train[train_cols], train['cnt'])

test_predictions = reg.predict(test[train_cols])
test_mse = mean_squared_error(test_predictions, test['cnt'])

test_mse


# This error is very high. Based on our histogram above, this could be due by having a few very high counts. 

# ## Decision Tree Algorithm
# 
# After applying the decision tree alogrithm, I will compare its error with the error from linear regression. I expect the decision tree error to be better since decision trees tend to predict outcomes much more reliably than linear regression models. 

# In[38]:


from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor(random_state = 1, min_samples_leaf = 5)
clf.fit(train[train_cols], train['cnt'])
predictions = clf.predict(test[train_cols])

np.mean((predictions - test['cnt'])**2)


# The error is certainly better than the linear regression model

# ## Random Forest Algorithm
# 
# I will now apply the random forest algorithm, whish should improve on the decision tree algorithm

# In[40]:


from sklearn.ensemble import RandomForestRegressor
rfor = RandomForestRegressor(random_state=1, min_samples_leaf = 5)
rfor.fit(train[train_cols], train['cnt'])
predictions = rfor.predict(test[train_cols])

np.mean((predictions - test['cnt'])**2)


# The random forest algorithm has the smallest error. 
