#!/usr/bin/env python
# coding: utf-8

# # Is Fandango Still Inflating Ratings?
# 
# In this project, I will use Python to help determine if Fandango still inflates their movie ratings. In 2015, [538](https://fivethirtyeight.com/features/fandango-movies-ratings/) showed that movie ratings on Fandango were higher than ratings on other sites. Fandango promised to fix this, and I wish to see if this fix occurred. 

# ## Importing the Data
# 
# We will work with two data sets for this project. First, we will use 538's [original data set](https://github.com/fivethirtyeight/data/tree/master/fandango) and then compare that with movies that [debuted in 2016 and 2017](https://github.com/mircealex/Movie_ratings_2016_17). 

# In[2]:


import pandas as pd

previous = pd.read_csv('fandango_score_comparison.csv')
after = pd.read_csv('movie_ratings_16_17.csv')

previous.head(3)


# In[3]:


fandango_previous = previous[['FILM', 'Fandango_Stars', 'Fandango_Ratingvalue', 'Fandango_votes',
                             'Fandango_Difference']].copy()
fandango_after = after[['movie', 'year', 'fandango']].copy()

fandango_previous.head(3)


# In the code above, I isolated the columns that offer information about Fandango's ratings. 
# 
# For the new data set, I did the same. 

# In[4]:


fandango_after.head(3)


# Again, I wish to know whether the movies after 2015 are still inflated. I will take two samples: before 2016 and during/after 2016. 
# 
# However, after reading the README documents from each data set, each set was sampled with different strategies, so a direct comparision between the two would be unwise as the sampling processes were not random. 
# 
# So, I have two choices: I can either collect new data, or change the goal of my analysis. I will choose the latter. 

# ## New Goal
# 
# I want to know whether there is any difference in Fandango ratings in 2015 vs 2016 for popular movies only. 
# 
# I will define a movie as "popular" if it has at least 30 fan ratings. 
# 
# Because of the limitations of the data, I will have to manually determine if the movies in the new data set are popular. Rather than checking every film, I will take a sample of the data set and inspect Fandango's website. 

# In[5]:


fandango_after.sample(10, random_state = 1)


# After checking each film in the sample above on Fandango's website, the only film with less than 30 ratings was "Cell". As 90% of the movies were popular, I will continue on with my analysis. 
# 
# I will now isolate the two data sets: the first data set will be popular movies released in 2016, while the second data set will be popular movies released in 2016. 

# In[5]:


fandango_previous['Year'] = fandango_previous['FILM'].str[-5:-1]
fandango_2015 = fandango_previous[fandango_previous['Year'] == '2015'].copy()
print(fandango_2015["Year"].value_counts())
print('/n')
fandango_2016 = fandango_after[fandango_after['year'] == 2016].copy()
print(fandango_2016["year"].value_counts())


# ## Comparing the distributions
# 
# I will now create two density plots of the distributions to begin visualization. 

# In[7]:


import matplotlib.pyplot as plt
from numpy import arange
get_ipython().magic('matplotlib inline')
plt.style.use('fivethirtyeight')


fandango_2015['Fandango_Stars'].plot.kde(label = '2015', legend = True, figsize = (8,5.5))
fandango_2016['fandango'].plot.kde(label = '2016', legend = True)

plt.title("Comparing distribution shapes for Fandango's ratings\n(2015 vs 2016)",
          y = 1.07)
plt.xlabel('Number of Stars')
plt.xlim(0,5)
plt.xticks(arange(0,5.1,.5))
plt.show()


# We can see that the 2015 data is more left-skewed than the 2016 data. This provides evidence that Fandango did indeed adjust their ratings after the 538 article was published. 
# 
# The frequency tables below show the same effect. 

# In[8]:


fandango_2015['Fandango_Stars'].value_counts(normalize = True).sort_index()
## 2015 data


# In[10]:


fandango_2016['fandango'].value_counts(normalize = True).sort_index()
## 2016 data


# We can also see this difference when we compute the mean, median, and mode for both sets of data. 

# In[16]:


mean_2015 = fandango_2015["Fandango_Stars"].mean()
median_2015 = fandango_2015["Fandango_Stars"].median()
mode_2015 = fandango_2015["Fandango_Stars"].mode()[0]

mean_2016 = fandango_2016["fandango"].mean()
median_2016 = fandango_2016["fandango"].median()
mode_2016 = fandango_2016["fandango"].mode()[0]

summary = pd.DataFrame()
summary['2015'] = [mean_2015, median_2015, mode_2015]
summary['2016'] = [mean_2016, median_2016, mode_2016]
summary.index = ['mean', 'median', 'mode']
summary


# In[17]:


plt.style.use('fivethirtyeight')
summary['2015'].plot.bar(color = '#0000FF', align = 'center', label = '2015', width = .25)
summary['2016'].plot.bar(color = '#FF0000', align = 'edge', label = '2016', width = .25,
                         rot = 0, figsize = (8,5))

plt.title('Comparing summary statistics: 2015 vs 2016', y = 1.07)
plt.ylim(0,5.5)
plt.yticks(arange(0,5.1,.5))
plt.ylabel('Stars')
plt.legend(framealpha = 0, loc = 'upper center')
plt.show()

