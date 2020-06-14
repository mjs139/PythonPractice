#!/usr/bin/env python
# coding: utf-8

# # Find the Best Markets For A E-Learning Product
# 
# In this project, I will use Python to help determine the best markets for e-learning, specializing in programming. 

# ## Understanding the Data
# 
# [Free Code Camp](https://www.freecodecamp.org/news/we-asked-20-000-people-who-they-are-and-how-theyre-learning-to-code-fff5d668969/) provided a survey in 2017 regarding interests of new coders. Their data can be found [here](https://github.com/freeCodeCamp/2017-new-coder-survey). 

# In[6]:


import pandas as pd

fcc = pd.read_csv('https://raw.githubusercontent.com/freeCodeCamp/2017-new-coder-survey/master/clean-data/2017-fCC-New-Coders-Survey-Data.csv', low_memory = 0)
fcc.shape


# In[9]:


fcc.head()


# ## Sample Representivity
# 
# I first wish to see if this survey is representative of the people I wish to capture: people interested in beginning to code. The HobRoleInterest column would be helpeful in determining this. I will explore this column further. 

# In[14]:


fcc["JobRoleInterest"].value_counts(normalize = True).sort_values(ascending = False)


# It looks like the survey participants can be interested in more than one role. The most popular roles are web developers and data scientists. 
# 
# For this project, I want to focus on web and mobile development. So, I wish to know what percentage of survey respondents are interested in at least one of these two subjects. 

# In[16]:


# Frequency table
interests_no_nulls = fcc['JobRoleInterest'].dropna()
web_or_mobile = interests_no_nulls.str.contains(
    'Web Developer|Mobile Developer') # returns an array of booleans
freq_table = web_or_mobile.value_counts(normalize = True)
print(freq_table)

# Graph for the frequency table above
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

freq_table.plot.bar()
plt.title('Most Participants are Interested in \nWeb or Mobile Development',
          y = 1.08) # y pads the title upward
plt.ylabel('Percentage', fontsize = 12)
plt.xticks([0,1],['Web or mobile\ndevelopment', 'Other subject'],
           rotation = 0) # the initial xtick labels were True and False
plt.ylim([0,1])
plt.show()


# It looks like roughly 86% of survey participants were interested in web or mobile development. 

# ## Finding Locations
# 
# I now wish to find where these new coders are located, and the densities of new coders in different locations. To accomplish this, I will look at the column CountryLive. 

# In[20]:


fcc_good = fcc[fcc['JobRoleInterest'].notnull()].copy()
fcc_good["CountryLive"].value_counts(normalize = True).sort_values(ascending = False).head()


# It looks like the United States is the largest market for this survey. 

# ## Money Spent
# 
# I now wish to see how much money a new coder would actually spend on learning. The MoneyForLearning column will be helpful for this. I will narrow my analysis for only the USA, India, UK, and Canada as they are the in the top four countries of my previoius table and all have a large percentage of the population who speak English. 

# In[23]:


fcc_good['MonthsProgramming'].replace(0,1, inplace = True)
#replacing zeroes with ones to avoid divide by zero error

fcc_good['money_per_month'] = fcc_good['MoneyForLearning'] / fcc_good['MonthsProgramming']
fcc_good['money_per_month'].isnull().sum()


# In[24]:


# Keep only the rows with non-nulls in the `money_per_month` column 
fcc_good = fcc_good[fcc_good['money_per_month'].notnull()]


# In[26]:


fcc_good = fcc_good[fcc_good["CountryLive"].notnull()]


# In[29]:


countries_mean = fcc_good.groupby('CountryLive').mean()
countries_mean['money_per_month'][['United States of America',
                            'India', 'United Kingdom',
                            'Canada']]


# It looks like those living in the United States are willing to pay significantly more per month to learn how to code. I will inspect this data further with box plots. 

# In[30]:


# Isolate only the countries of interest
only_4 = fcc_good[fcc_good['CountryLive'].str.contains(
    'United States of America|India|United Kingdom|Canada')]

# Box plots to visualize distributions
import seaborn as sns
sns.boxplot(y = 'money_per_month', x = 'CountryLive',
            data = only_4)
plt.title('Money Spent Per Month Per Country\n(Distributions)',
         fontsize = 16)
plt.ylabel('Money per month (US dollars)')
plt.xlabel('Country')
plt.xticks(range(4), ['US', 'UK', 'India', 'Canada']) # avoids tick labels overlap
plt.show()


# I see some extreme outliers for each country. I now wish to eliminate these. I will eliminate all data greater than $500 per month. 

# In[32]:


only_4_good = only_4[only_4["money_per_month"] < 500]

sns.boxplot(y = 'money_per_month', x = 'CountryLive',
            data = only_4_good)
plt.title('Money Spent Per Month Per Country\n(Distributions)',
         fontsize = 16)
plt.ylabel('Money per month (US dollars)')
plt.xlabel('Country')
plt.xticks(range(4), ['US', 'UK', 'India', 'Canada']) # avoids tick labels overlap
plt.show()


# In[33]:


only_4_good.groupby('CountryLive').mean()['money_per_month']


# With some outliers removes, we see that most customers in these four countries would be willing to pay roughly $25 per month to learn how to code. 
