
# Analyzing Jeopardy Questions

In this Python project, I will work with a dataset of Jeopardy questions to figure out some patterns. The [dataset](https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file) contains 20,000 rows filled with questions. 

## Exploring the Data


```python
import pandas as pd

jeopardy = pd.read_csv("jeopardy.csv")
print(jeopardy.shape)
jeopardy.head()
```

    (19999, 7)





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
      <th>Show Number</th>
      <th>Air Date</th>
      <th>Round</th>
      <th>Category</th>
      <th>Value</th>
      <th>Question</th>
      <th>Answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>HISTORY</td>
      <td>$200</td>
      <td>For the last 8 years of his life, Galileo was ...</td>
      <td>Copernicus</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>ESPN's TOP 10 ALL-TIME ATHLETES</td>
      <td>$200</td>
      <td>No. 2: 1912 Olympian; football star at Carlisl...</td>
      <td>Jim Thorpe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EVERYBODY TALKS ABOUT IT...</td>
      <td>$200</td>
      <td>The city of Yuma in this state has a record av...</td>
      <td>Arizona</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>THE COMPANY LINE</td>
      <td>$200</td>
      <td>In 1963, live on "The Art Linkletter Show", th...</td>
      <td>McDonald's</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EPITAPHS &amp; TRIBUTES</td>
      <td>$200</td>
      <td>Signer of the Dec. of Indep., framer of the Co...</td>
      <td>John Adams</td>
    </tr>
  </tbody>
</table>
</div>




```python
jeopardy.columns
```




    Index(['Show Number', ' Air Date', ' Round', ' Category', ' Value',
           ' Question', ' Answer'],
          dtype='object')



Some of the column names have spaces in front. I will remove spaces in each item in jeopardy.columns.


```python
jeopardy.columns = ['Show Number', 'Air Date', 'Round', 'Category', 'Value', 'Question', 'Answer']
jeopardy.columns
```




    Index(['Show Number', 'Air Date', 'Round', 'Category', 'Value', 'Question',
           'Answer'],
          dtype='object')



## Normalizing the Text Columns

Before doing any analysis, I need to normalize all of the test columns. I want to ensure lowercase words and no punctuation. 


```python
import re

def norm(text):
    text = text.lower()
    text = re.sub("[^A-Za-z0-9\s]", "", text)
    text = re.sub("\s+", " ", text)
    return text

jeopardy["clean_question"] = jeopardy["Question"].apply(norm)
jeopardy["clean_answer"] = jeopardy["Answer"].apply(norm)
jeopardy.head()
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
      <th>Show Number</th>
      <th>Air Date</th>
      <th>Round</th>
      <th>Category</th>
      <th>Value</th>
      <th>Question</th>
      <th>Answer</th>
      <th>clean_question</th>
      <th>clean_answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>HISTORY</td>
      <td>$200</td>
      <td>For the last 8 years of his life, Galileo was ...</td>
      <td>Copernicus</td>
      <td>for the last 8 years of his life galileo was u...</td>
      <td>copernicus</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>ESPN's TOP 10 ALL-TIME ATHLETES</td>
      <td>$200</td>
      <td>No. 2: 1912 Olympian; football star at Carlisl...</td>
      <td>Jim Thorpe</td>
      <td>no 2 1912 olympian football star at carlisle i...</td>
      <td>jim thorpe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EVERYBODY TALKS ABOUT IT...</td>
      <td>$200</td>
      <td>The city of Yuma in this state has a record av...</td>
      <td>Arizona</td>
      <td>the city of yuma in this state has a record av...</td>
      <td>arizona</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>THE COMPANY LINE</td>
      <td>$200</td>
      <td>In 1963, live on "The Art Linkletter Show", th...</td>
      <td>McDonald's</td>
      <td>in 1963 live on the art linkletter show this c...</td>
      <td>mcdonalds</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EPITAPHS &amp; TRIBUTES</td>
      <td>$200</td>
      <td>Signer of the Dec. of Indep., framer of the Co...</td>
      <td>John Adams</td>
      <td>signer of the dec of indep framer of the const...</td>
      <td>john adams</td>
    </tr>
  </tbody>
</table>
</div>



I will now normalize the Value and Air Date columns. The value column should be numeric. The Air Date columns should be a datetime, not a string


```python
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
      <th>Show Number</th>
      <th>Air Date</th>
      <th>Round</th>
      <th>Category</th>
      <th>Value</th>
      <th>Question</th>
      <th>Answer</th>
      <th>clean_question</th>
      <th>clean_answer</th>
      <th>clean_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>HISTORY</td>
      <td>$200</td>
      <td>For the last 8 years of his life, Galileo was ...</td>
      <td>Copernicus</td>
      <td>for the last 8 years of his life galileo was u...</td>
      <td>copernicus</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>ESPN's TOP 10 ALL-TIME ATHLETES</td>
      <td>$200</td>
      <td>No. 2: 1912 Olympian; football star at Carlisl...</td>
      <td>Jim Thorpe</td>
      <td>no 2 1912 olympian football star at carlisle i...</td>
      <td>jim thorpe</td>
      <td>200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EVERYBODY TALKS ABOUT IT...</td>
      <td>$200</td>
      <td>The city of Yuma in this state has a record av...</td>
      <td>Arizona</td>
      <td>the city of yuma in this state has a record av...</td>
      <td>arizona</td>
      <td>200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>THE COMPANY LINE</td>
      <td>$200</td>
      <td>In 1963, live on "The Art Linkletter Show", th...</td>
      <td>McDonald's</td>
      <td>in 1963 live on the art linkletter show this c...</td>
      <td>mcdonalds</td>
      <td>200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EPITAPHS &amp; TRIBUTES</td>
      <td>$200</td>
      <td>Signer of the Dec. of Indep., framer of the Co...</td>
      <td>John Adams</td>
      <td>signer of the dec of indep framer of the const...</td>
      <td>john adams</td>
      <td>200</td>
    </tr>
  </tbody>
</table>
</div>



## Initial Explorations

Before proceeding, I wish to figure out two things:
- How often the answer is deducible from the question
- How often new questions are repeats of older questions

I will answer the former by seeing how many times words in the answer occur in the question. 

For the latter, I will see how often complex words (> 6 characters) reoccur. 


```python
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
```




    0.05900196524977763



It looks like 6% of Jeopardy questions include a word that is also in the answer. 

Now I will work on the repeating questions question. One problem is that I only have roughly 10% of the full Jeopardy question dataset. 


```python
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
```




    0.6876260592169802



It looks like 68% of questions show signs of being recycled. 

## Chi Squared Tests

I want to figure out which terms correspond to high-value questions using a chi-squared test. I will first need to narrow down the questions into two categories:

- low value (< 800)
- high value (> 800)


```python
#determines whether a question is high or low value
def determine_value(row):
    value = 1
    if row["clean_value"] > 800:
        value = 1
    else:
        value = 0
    return value
```


```python
jeopardy["high_value"] = jeopardy.apply(determine_value, axis=1)
jeopardy.head()
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
      <th>Show Number</th>
      <th>Air Date</th>
      <th>Round</th>
      <th>Category</th>
      <th>Value</th>
      <th>Question</th>
      <th>Answer</th>
      <th>clean_question</th>
      <th>clean_answer</th>
      <th>clean_value</th>
      <th>answer_in_question</th>
      <th>question_overlap</th>
      <th>high_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19325</th>
      <td>10</td>
      <td>1984-09-21</td>
      <td>Final Jeopardy!</td>
      <td>U.S. PRESIDENTS</td>
      <td>None</td>
      <td>Adventurous 26th president, he was 1st to ride...</td>
      <td>Theodore Roosevelt</td>
      <td>adventurous 26th president he was 1st to ride ...</td>
      <td>theodore roosevelt</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19301</th>
      <td>10</td>
      <td>1984-09-21</td>
      <td>Double Jeopardy!</td>
      <td>LABOR UNIONS</td>
      <td>$200</td>
      <td>Notorious labor leader missing since '75</td>
      <td>Jimmy Hoffa</td>
      <td>notorious labor leader missing since 75</td>
      <td>jimmy hoffa</td>
      <td>200</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19302</th>
      <td>10</td>
      <td>1984-09-21</td>
      <td>Double Jeopardy!</td>
      <td>1789</td>
      <td>$200</td>
      <td>Washington proclaimed Nov. 26, 1789 this first...</td>
      <td>Thanksgiving</td>
      <td>washington proclaimed nov 26 1789 this first n...</td>
      <td>thanksgiving</td>
      <td>200</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19303</th>
      <td>10</td>
      <td>1984-09-21</td>
      <td>Double Jeopardy!</td>
      <td>TOURIST TRAPS</td>
      <td>$200</td>
      <td>Both Ferde Grofe' &amp; the Colorado River dug thi...</td>
      <td>the Grand Canyon</td>
      <td>both ferde grofe the colorado river dug this n...</td>
      <td>the grand canyon</td>
      <td>200</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19304</th>
      <td>10</td>
      <td>1984-09-21</td>
      <td>Double Jeopardy!</td>
      <td>LITERATURE</td>
      <td>$200</td>
      <td>Depending on the book, he could be a "Jones", ...</td>
      <td>Tom</td>
      <td>depending on the book he could be a jones a sa...</td>
      <td>tom</td>
      <td>200</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```


```python
from random import choice

terms_used_list = list(terms_used)
comparison_terms = [choice(terms_used_list) for _ in range(10)]

observed_expected = []

for term in comparison_terms:
    high_low = count_usage(term)
    observed_expected.append(high_low)
    
observed_expected
```




    [(1, 0),
     (2, 6),
     (1, 0),
     (1, 0),
     (1, 2),
     (0, 2),
     (1, 0),
     (3, 5),
     (1, 3),
     (0, 2)]



NOw that I have found the observed counts for a few terms, I can compute the expected counts and the chi-squared value


```python
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
```




    [Power_divergenceResult(statistic=2.487792117195675, pvalue=0.11473257634454047),
     Power_divergenceResult(statistic=0.05272886616881538, pvalue=0.818381104912348),
     Power_divergenceResult(statistic=2.487792117195675, pvalue=0.11473257634454047),
     Power_divergenceResult(statistic=2.487792117195675, pvalue=0.11473257634454047),
     Power_divergenceResult(statistic=0.03188116723440362, pvalue=0.8582887163235293),
     Power_divergenceResult(statistic=0.803925692253768, pvalue=0.3699222378079571),
     Power_divergenceResult(statistic=2.487792117195675, pvalue=0.11473257634454047),
     Power_divergenceResult(statistic=0.30490002599164673, pvalue=0.5808267268567516),
     Power_divergenceResult(statistic=0.02636443308440769, pvalue=0.871013484688921),
     Power_divergenceResult(statistic=0.803925692253768, pvalue=0.3699222378079571)]



The results show that no terms had a significant different in usage between high value and low value rows. 
