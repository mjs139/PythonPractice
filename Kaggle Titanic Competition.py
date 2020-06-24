#!/usr/bin/env python
# coding: utf-8

# # Kaggle Titanic Competition
# 
# In this project, I will compete in the Kaggle Titanic Competition. 

# In[12]:


import pandas as pd

train = pd.read_csv("train.csv")
holdout = pd.read_csv("test.csv")

print(train.shape)
train.head()


# ## Loading Functions

# In[13]:


# %load functions.py
def process_missing(df):
    """Handle various missing values from the data set

    Usage
    ------

    holdout = process_missing(holdout)
    """
    df["Fare"] = df["Fare"].fillna(train["Fare"].mean())
    df["Embarked"] = df["Embarked"].fillna("S")
    return df

def process_age(df):
    """Process the Age column into pre-defined 'bins' 

    Usage
    ------

    train = process_age(train)
    """
    df["Age"] = df["Age"].fillna(-0.5)
    cut_points = [-1,0,5,12,18,35,60,100]
    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

def process_fare(df):
    """Process the Fare column into pre-defined 'bins' 

    Usage
    ------

    train = process_fare(train)
    """
    cut_points = [-1,12,50,100,1000]
    label_names = ["0-12","12-50","50-100","100+"]
    df["Fare_categories"] = pd.cut(df["Fare"],cut_points,labels=label_names)
    return df

def process_cabin(df):
    """Process the Cabin column into pre-defined 'bins' 

    Usage
    ------

    train process_cabin(train)
    """
    df["Cabin_type"] = df["Cabin"].str[0]
    df["Cabin_type"] = df["Cabin_type"].fillna("Unknown")
    df = df.drop('Cabin',axis=1)
    return df

def process_titles(df):
    """Extract and categorize the title from the name column 

    Usage
    ------

    train = process_titles(train)
    """
    titles = {
        "Mr" :         "Mr",
        "Mme":         "Mrs",
        "Ms":          "Mrs",
        "Mrs" :        "Mrs",
        "Master" :     "Master",
        "Mlle":        "Miss",
        "Miss" :       "Miss",
        "Capt":        "Officer",
        "Col":         "Officer",
        "Major":       "Officer",
        "Dr":          "Officer",
        "Rev":         "Officer",
        "Jonkheer":    "Royalty",
        "Don":         "Royalty",
        "Sir" :        "Royalty",
        "Countess":    "Royalty",
        "Dona":        "Royalty",
        "Lady" :       "Royalty"
    }
    extracted_titles = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    df["Title"] = extracted_titles.map(titles)
    return df

def create_dummies(df,column_name):
    """Create Dummy Columns (One Hot Encoding) from a single Column

    Usage
    ------

    train = create_dummies(train,"Age")
    """
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df


# In[14]:


def process(df):
    df = process_missing(df)
    df = process_age(df)
    df = process_fare(df)
    df = process_titles(df)
    df = process_cabin(df)
    for col in ["Age_categories","Fare_categories",
                "Title","Cabin_type","Sex"]:
        df = create_dummies(df,col)
    
    return df 


# In[15]:


train = process(train)
holdout = process(holdout)

print(train.shape)
train.head()


# ## Exploring the Data
# 
# I am going to examine the two columns that contain information about the family members each passenger has onboard: SibSp and Parch. These columns count the number of siblings/spouses and the number of parents/children each passenger has onboard respectively. 

# In[17]:


explore = train[["SibSp", "Parch", "Survived"]].copy()
explore.info()


# In[18]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

explore.drop("Survived",axis=1).plot.hist(alpha=0.5,bins=8)
plt.show()


# We can see that our data is heavily skewed right. 
# 
# I will combine these columns and look at the resulting distribution of values and survival rate

# In[20]:


explore["combined"] = explore["SibSp"] + explore["Parch"]

explore.drop("Survived",axis=1).plot.hist(alpha=0.5,bins=10)
plt.xticks(range(11))
plt.show()


# In[22]:


import numpy as np

for col in explore.columns.drop("Survived"):
    pivot = explore.pivot_table(index=col,values="Survived")
    pivot.plot.bar(ylim=(0,1),yticks=np.arange(0,1,.1))
    plt.axhspan(.3, .6, alpha=0.2, color='red')
    plt.show()


# We can see that if you had no family member onboard, you were more likely to die. 
# 
# Based off this, I will create a new feature: was the passenger alone. This will be a binary column with a 1 representing that the passenger had no family members onboard. 

# In[46]:


def isalone(df):
    df["isalone"] = 0
    df["combined"] = df["SibSp"] + df["Parch"]
    bool_values = df["combined"] == 0
    df.loc[bool_values, "isalone"] = 1
    df = df.drop("combined", axis = 1)
    return df


# In[47]:


train = isalone(train)
holdout = isalone(holdout)

train.head()


# ## Feature Selection
# 
# I will create a function that automates selecting the best performing features using recursive feature elimination using Random Forest algorithm. 

# In[48]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

def select_features(df):
    #remove non-numeric columns and null values
    df = df.select_dtypes([np.number]).dropna(axis=1)
    
    all_X = df.drop(["PassengerId", "Survived"], axis = 1)
    all_y = df["Survived"]
    
    clf = RandomForestClassifier(random_state=1)
    selector = RFECV(clf,cv=10)
    selector.fit(all_X,all_y)
    
    best_columns = list(all_X.columns[selector.support_])
    print("Best Columns \n"+"-"*12+"\n{}\n".format(best_columns))
    
    return best_columns


# In[49]:


best_cols = select_features(train)


# In[50]:


len(best_cols)


# ## Model Selection and Tuning
# 
# I will write a function to do the heavy lifting of model selection and tuning. The function will use three different algorithms and use grid search to train using different combinations of hyperparamters to find the best performing model. 
# 
# I will return a list of dictionaries to see which was the most accurate. 

# In[61]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def select_model(df, features):
    all_X = df[features]
    all_y = df["Survived"]
    
    models = [
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(),
            "hyperparameters":
                {
                    "solver": ["newton-cg", "lbfgs", "liblinear"]
                }
        },
        {
            "name": "KNeighborsClassifier",
            "estimator": KNeighborsClassifier(),
            "hyperparameters":
                {
                    "n_neighbors": range(1,20,2),
                    "weights": ["distance", "uniform"],
                    "algorithm": ["ball_tree", "kd_tree", "brute"],
                    "p": [1,2]
                }
        },
        {
            "name": "RandomForestClassifier",
            "estimator": RandomForestClassifier(random_state=1),
            "hyperparameters":
                {
                    "n_estimators": [4, 6, 9],
                    "criterion": ["entropy", "gini"],
                    "max_depth": [2, 5, 10],
                    "max_features": ["log2", "sqrt"],
                    "min_samples_leaf": [1, 5, 8],
                    "min_samples_split": [2, 3, 5]

                }
        }
    ]    
    for model in models:
        print(model["name"])
        print('\n')
        
        grid = GridSearchCV(model["estimator"], param_grid = model["hyperparameters"], cv = 10)
        grid.fit(all_X, all_y)
        model["best_params"] = grid.best_params_
        model["best_score"] = grid.best_score_
        model["best_model"] = grid.best_estimator_

        print("Best Score: {}".format(model["best_score"]))
        print("Best Parameters: {}\n".format(model["best_params"]))
    return models


# In[62]:


result = select_model(train,best_cols)


# ## Automating Kaggle Submission
# 
# I will now create a function to automate my Kaggle submissions

# In[66]:


def save_submission_file(model, cols, filename="submission.csv"):
    holdout_data = holdout[cols]
    predictions = model.predict(holdout_data)
    
    holdout_ids = holdout["PassengerId"]
    submission_df = {"PassengerId": holdout_ids,
                 "Survived": predictions}
    submission = pd.DataFrame(submission_df)

    submission.to_csv(filename,index=False)

best_rf_model = result[2]["best_model"]
save_submission_file(best_rf_model,best_cols)

