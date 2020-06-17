#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbors and A Car's Market Price
# 
# In this project, I will practice the machine learning workflow to predict a car's market price using its attributes. The dataset found at the (UCI Machine Learning Repository)[https://archive.ics.uci.edu/ml/datasets/automobile] contains information about the technical aspects of different vehicles. 

# ## Exploring the data

# In[10]:


import pandas as pd

cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('imports-85.data', names=cols)

print(cars.shape)
cars.head()


# In[11]:


# Select only the columns with continuous values from - https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
numeric_cars = cars[continuous_values_cols]

numeric_cars.head()


# ## Data Cleaning
# 
# In order to prepare the data for predictive modeling, I need to make sure that there are no missing data values. First I will replace all of the ? values with the numpy.nan missing value. Then, I will change columns from the pandas object data type to a numeric type. 

# In[12]:


import numpy as np

numeric_cars = numeric_cars.replace("?", np.nan)

numeric_cars = numeric_cars.astype('float')
numeric_cars.isnull().sum()


# Since I want to predict price, I will remove any rows with a missing price value

# In[13]:


numeric_cars = numeric_cars.dropna(subset=['price'])
numeric_cars.isnull().sum()


# For the remaining missing values, I will replace them with the mean of each column. 

# In[14]:


numeric_cars = numeric_cars.fillna(numeric_cars.mean())
numeric_cars.isnull().sum()


# I will now normalize all of the columns, except for price, so that the values range from 0 to 1

# In[15]:


price_col = numeric_cars['price']
numeric_cars = (numeric_cars - numeric_cars.min())/(numeric_cars.max() - numeric_cars.min())
numeric_cars['price'] = price_col

numeric_cars.head(2)


# ## Univariate Model
# 
# I will start with simple univariate models before moving to more complex models. 

# In[17]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def knn_train_test(train_name, target_name, df):
    # randomizing the data
    np.random.seed(42)
    shuf = np.random.permutation(len(df))
    shuf_df = df.iloc[shuf]
    
    # Divide number of rows in half and round.
    last_train_row = int(len(shuf_df) / 2)
    
    # creating a training set and test set
    train_df = shuf_df.iloc[:last_train_row]
    test_df = shuf_df.iloc[last_train_row:]
    
    # Fit a model on the training set
    knn = KNeighborsRegressor()
    train_features = train_df[[train_name]]
    train_target = train_df[target_name]
    knn.fit(train_features, train_target)
    
    predictions = knn.predict(test_df[[train_name]])
    
    # Calculate RMSE and return it
    mse = mean_squared_error(test_df[target_name], predictions)
    rmse = mse ** (1/2)
    return rmse


# I will use this function to train and test univariate models using the different numeric columns in the data set, trying to find the one with the best results

# In[18]:


rmse_results = {}
train_cols = numeric_cars.columns.drop('price')

for col in train_cols:
    rmse_val = knn_train_test(col, 'price', numeric_cars)
    rmse_results[col] = rmse_val

rmse_results_series = pd.Series(rmse_results)
rmse_results_series.sort_values()


# It looks like engine size offers the best results. 
# 
# I will now modify the knn_train_test function to accept a paramter for the k-value. For each numeric column, I will create, train, and test a univariate model using the following k-values (1, 3, 5, 7, 9). I will visualize the results using a scatter plot. 

# In[22]:


def knn_train_test_mod(train_name, target_name, df):
    # randomizing the data
    np.random.seed(42)
    shuf = np.random.permutation(len(df))
    shuf_df = df.iloc[shuf]
    
    # Divide number of rows in half and round.
    last_train_row = int(len(shuf_df) / 2)
    
    # creating a training set and test set
    train_df = shuf_df.iloc[:last_train_row]
    test_df = shuf_df.iloc[last_train_row:]
    
    k_values = [1,3,5,7,9]
    k_rmses = {}
    
    # Fit a model on the training set
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        train_features = train_df[[train_name]]
        train_target = train_df[target_name]
        knn.fit(train_features, train_target)
    
        predictions = knn.predict(test_df[[train_name]])

        # Calculate RMSE and return it
        mse = mean_squared_error(test_df[target_name], predictions)
        rmse = mse ** (1/2)
        
        k_rmses[k] = rmse
    return k_rmses


# In[24]:


k_rmse_results = {}

train_cols = numeric_cars.columns.drop('price')
for col in train_cols:
    rmse_val = knn_train_test_mod(col, 'price', numeric_cars)
    k_rmse_results[col] = rmse_val

k_rmse_results


# In[28]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x,y)
    plt.xlabel('k-value')
    plt.ylabel('RMSE')


# ## Multivariate Model
# 
# I will now modeify the kmm_train_test function to work with multiple columns. 

# In[31]:


# Compute average RMSE across different `k` values for each feature.
feature_avg_rmse = {}
for k,v in k_rmse_results.items():
    avg_rmse = np.mean(list(v.values()))
    feature_avg_rmse[k] = avg_rmse
series_avg_rmse = pd.Series(feature_avg_rmse)
sorted_series_avg_rmse = series_avg_rmse.sort_values()
print(sorted_series_avg_rmse)

sorted_features = sorted_series_avg_rmse.index


# In[32]:


def knn_train_test(train_cols, target_col, df):
    np.random.seed(1)
    
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [5]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[train_cols])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses


# In[33]:


for nr_best_feats in range(2,7):
    k_rmse_results['{} best features'.format(nr_best_feats)] = knn_train_test(
        sorted_features[:nr_best_feats],
        'price',
        numeric_cars
    )

k_rmse_results


# ## Hyperparameter Tuning

# In[34]:


def knn_train_test(train_cols, target_col, df):
    np.random.seed(1)
    
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [i for i in range(1, 25)]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[train_cols])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses


# In[35]:


k_rmse_results = {}

for nr_best_feats in range(2,6):
    k_rmse_results['{} best features'.format(nr_best_feats)] = knn_train_test(
        sorted_features[:nr_best_feats],
        'price',
        numeric_cars
    )

k_rmse_results


# In[36]:


for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())  
    plt.plot(x,y, label="{}".format(k))
    
plt.xlabel('k value')
plt.ylabel('RMSE')
plt.legend()

