#!/usr/bin/env python
# coding: utf-8

# # Popular Data Science Questions in Python
# 
# In this project, I will use the Data Science Stack Exchange to help determine what content a data science course should focus on. To do this, I will use Stack Exchange's tagging system to determine popular topics. This data is available on the [Stack Exchange Data Explorer](https://data.stackexchange.com/datascience/query/new)

# I will focus my attention on the columns that seem most relevant to my goal: 
# - ID
# - PostTypeId
# - CreationDate
# - Score
# - ViewCount
# - Tags
# - AnswerCount
# - FavoriteCount

# ## Exploring the Data

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')


# In[2]:


questions = pd.read_csv("2019_questions.csv", parse_dates=["CreationDate"])


# In[4]:


questions.head()


# In[5]:


questions.info()


# FavoriteCount has a number of missing values. I will replace these with zeroes since they have not been favorited. 

# In[9]:


questions.fillna(value={"FavoriteCount": 0}, inplace=True)
questions["FavoriteCount"] = questions["FavoriteCount"].astype(int)
print(questions.dtypes)
print('\n')
print(questions.info())


# ## Tags Column Manipulation
# 
# I will now explore the tags column in more detail

# In[8]:


questions["Tags"].apply(lambda value: type(value)).unique()


# It looks like we have a columns of strings. Stack Exchange only allows al maximum of 5 tags, so I will break this into 5 columns: Tag1, Tag2, etc.
# 
# Also, the tags are separated by <>. I wish to change this separation to commas. 

# In[13]:


questions["Tags"] = questions["Tags"].str.replace("><",",").str.replace("<|>", "").str.split(",")


# In[14]:


questions.head()


# ## Determining The Most Popular Tags
# 
# For each tag, I'll count how many times the tage was used and how many times a question with that tag was viewed. 

# In[16]:


#count how many times a tage was used

tag_count = dict()

for tags in questions["Tags"]:
    for tag in tags:
        if tag in tag_count:
            tag_count[tag] += 1
        else:
            tag_count[tag] = 1

tag_count = pd.DataFrame.from_dict(tag_count, orient="index")
tag_count.rename(columns={0: "Count"}, inplace=True)
tag_count.head(10)

most_used = tag_count.sort_values(by="Count", ascending = False).head(10)
most_used


# In[18]:


most_used.plot(kind="barh", figsize=(16,8))


# I will now repeat this process for views

# In[19]:


#count how many times a tage was viewed

tag_view_count = dict()

for index, row in questions.iterrows():
    for tag in row['Tags']:
        if tag in tag_view_count:
            tag_view_count[tag] += row['ViewCount']
        else:
            tag_view_count[tag] = row['ViewCount']
            
tag_view_count = pd.DataFrame.from_dict(tag_view_count, orient="index")
tag_view_count.rename(columns={0: "ViewCount"}, inplace=True)

most_viewed = tag_view_count.sort_values(by="ViewCount", ascending = False).head(10)

most_viewed.plot(kind="barh", figsize=(16,8))


# It looks like for both views and tag counts, machine learning and python are the most popular. 

# In[ ]:




