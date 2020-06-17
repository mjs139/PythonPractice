#!/usr/bin/env python
# coding: utf-8

# # Analyzing Jeopardy Questions
# 
# In this Python project, I will work with a dataset of Jeopardy questions to figure out some patterns. The [dataset](https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file) contains 20,000 rows filled with questions. 

# ## Exploring the Data

# In[1]:


import pandas as pd

jeopardy = pd.read_csv("jeopardy.csv")
print(jeopardy.shape)
jeopardy.head()


# In[2]:


jeopardy.columns


# Some of the column names have spaces in front. I will remove spaces in each item in jeopardy.columns.

# In[4]:


jeopardy.columns = ['Show Number', 'Air Date', 'Round', 'Category', 'Value', 'Question', 'Answer']
jeopardy.columns


# ## Normalizing the Text Columns
# 
# Before doing any analysis, I need to normalize all of the test columns. I want to ensure lowercase words and no punctuation. 

# In[7]:


import re

def norm(text):
    text = text.lower()
    text = re.sub("[^A-Za-z0-9\s]", "", text)
    text = re.sub("\s+", " ", text)
    return text

jeopardy["clean_question"] = jeopardy["Question"].apply(norm)
jeopardy["clean_answer"] = jeopardy["Answer"].apply(norm)
jeopardy.head()


# I will now normalize the Value and Air Date columns. The value column should be numeric. The Air Date columns should be a datetime, not a string

# In[9]:


def norm_value(string):
    text = re.sub("[^A-Za-z0-9\s]", "", string)
    try:
        text = int(text)
    except Exception:
        text = 0
    return text

jeopardy["clean_value"] = jeopardy["Value"].apply(norm_value)
jeopardy["Air Date"] = pd.to_datetime(jeopardy["Air Date"])

jeopardy.head()


# ## Initial Explorations
# 
# Before proceeding, I wish to figure out two things:
# - How often the answer is deducible from the question
# - How often new questions are repeats of older questions
# 
# I will answer the former by seeing how many times words in the answer occur in the question. 
# 
# For the latter, I will see how often complex words (> 6 characters) reoccur. 

# In[10]:


def count_matches(row):
    split_answer = row["clean_answer"].split()
    split_question = row["clean_question"].split()
    match_count = 0
    
    if "the" in split_answer: #since "the" is not useful here
        split_answer.remove("the")
    if len(split_answer) == 0: #prevents division by 0 error
        return 0
    for item in split_answer:
        if item in split_question:
            match_count += 1
    return match_count / len(split_answer)

jeopardy["answer_in_question"] = jeopardy.apply(count_matches, axis=1)
jeopardy["answer_in_question"].mean()


# It looks like 6% of Jeopardy questions include a word that is also in the answer. 
# 
# Now I will work on the repeating questions question. One problem is that I only have roughly 10% of the full Jeopardy question dataset. 

# In[11]:


question_overlap = []
terms_used = set()

jeopardy = jeopardy.sort_values("Air Date")

for i, row in jeopardy.iterrows():
        split_question = row["clean_question"].split(" ")
        split_question = [q for q in split_question if len(q) > 5] #removing words that are less than 6 characters long
        match_count = 0
        for word in split_question:
            if word in terms_used:
                match_count += 1
        for word in split_question:
            terms_used.add(word)
        if len(split_question) > 0:
            match_count /= len(split_question)
        question_overlap.append(match_count)
        
jeopardy["question_overlap"] = question_overlap

jeopardy["question_overlap"].mean()


# It looks like 68% of questions show signs of being recycled. 

# ## Chi Squared Tests
# 
# I want to figure out which terms correspond to high-value questions using a chi-squared test. I will first need to narrow down the questions into two categories:
# 
# - low value (< 800)
# - high value (> 800)

# In[12]:


#determines whether a question is high or low value
def determine_value(row):
    value = 1
    if row["clean_value"] > 800:
        value = 1
    else:
        value = 0
    return value


# In[13]:


jeopardy["high_value"] = jeopardy.apply(determine_value, axis=1)
jeopardy.head()


# In[18]:


def count_usage(word):
    low_count = 0
    high_count = 0
    for i, row in jeopardy.iterrows():
        if term in row["clean_question"].split(" "):
            if row["high_value"] == 1:
                high_count += 1
            else:
                low_count += 1
    return high_count, low_count


# In[21]:


from random import choice

terms_used_list = list(terms_used)
comparison_terms = [choice(terms_used_list) for _ in range(10)]

observed_expected = []

for term in comparison_terms:
    high_low = count_usage(term)
    observed_expected.append(high_low)
    
observed_expected


# NOw that I have found the observed counts for a few terms, I can compute the expected counts and the chi-squared value

# In[24]:


from scipy.stats import chisquare
import numpy as np

high_value_count = jeopardy[jeopardy["high_value"] == 1].shape[0]
low_value_count = jeopardy[jeopardy["high_value"] == 0].shape[0]

chi_squared = []

for lis in observed_expected:
    total = sum(lis)
    total_prob = total / jeopardy.shape[0]
    expected_high = total_prob * high_value_count
    expected_low = total_prob * low_value_count
    
    observed = np.array([lis[0], lis[1]])
    expected = np.array([expected_high, expected_low])
    chi_squared.append(chisquare(observed, expected))
    
chi_squared


# The results show that no terms had a significant different in usage between high value and low value rows. 
