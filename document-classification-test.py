#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# Importing all important libraries needed
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit

import pickle


# In[2]:


# Dataset Reading from csv file
df1 = pd.read_csv("shuffled-full-set-hashed.csv",header=None)
df1.columns =['type', 'content'] 
df=df1.copy()


# In[3]:


# let's see the distribution of the data to different classes
df.groupby('type').count().transpose()


# In[4]:


# Removal of Null values
df=df[df.content.notnull()].reset_index(drop=True)

# Removal of spaces from beginning and end
df=df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Removal of duplicates
df=df.drop_duplicates().reset_index(drop=True)


# In[5]:


# Let's check whether the null values are removing any specific class or not
df1[df1.content.isnull()].reset_index(drop=True).groupby('type')[['type']].count().transpose()


# In[6]:


# Spliting the dataset in training and testing
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df, df.type, test_size=0.20, 
                                                                    random_state=1000, stratify=df['type'])


# In[7]:


# Calling diff. vectorizers
CountVectTokenizer = CountVectorizer()
hashingVectTokenizer = HashingVectorizer()
TfidfVectTokenizer = TfidfVectorizer()
TfidfTransfTokenizer = TfidfTransformer()


# In[8]:


# Cnvert raw text data to numeric data || Tokenizing
tfidf = TfidfVectTokenizer.fit_transform(X_train.content.values.astype('U'))
tfidf1 = TfidfVectTokenizer.transform(X_test.content.values.astype('U'))


# In[9]:


# Building the Logistic Regression Model
model = LogisticRegression(n_jobs=-1, max_iter=50)
model.fit(tfidf, Y_train)


# In[10]:


# Evaluation || Accuracy
print(model.score(tfidf1, Y_test))


# In[11]:


# Store model and vectorizer
pickle.dump(model, open('MLmodel.pkl', 'wb'))
pickle.dump(TfidfVectTokenizer, open('vectorizor.pkl', 'wb'))

