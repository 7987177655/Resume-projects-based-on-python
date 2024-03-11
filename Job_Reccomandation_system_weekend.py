#!/usr/bin/env python
# coding: utf-8

# # Project - 7 ( Job Reccomendation System )

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from google.colab import files
path = list(files.upload().keys())[0]


# In[3]:


df = pd.read_csv(path)


# In[4]:


df.head()


# In[6]:


df.shape


# In[7]:


df = df.sample(5000 , ignore_index = True)


# In[9]:


import string


# In[10]:


panch = string.punctuation
panch


# In[11]:


def remove_panch(text):
  mylist = []
  for i in text.split():
    if i not in panch:
      mylist.append(i)
  return " ".join(mylist)


# In[12]:


df['Job Title'] = df['Job Title'].apply(remove_panch)


# In[13]:


df['Location'] = df['Location'].apply(remove_panch)


# In[15]:


# df['Company Name'] = df['Company Name'].apply(remove_panch)   #this code shows error because string method will not apply on float data .


# In[16]:


df['Company Name'] = df['Company Name'].astype(str)


# In[17]:


df['Company Name'] = df['Company Name'].apply(remove_panch)


# In[18]:


df['Content'] = df.apply(lambda x :" ".join(x.dropna().astype(str)) , axis=1)


# In[19]:


df


# In[20]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# In[21]:


cv = CountVectorizer(max_features = 5000)


# In[22]:


metrics = cv.fit_transform(df['Content']).toarray()


# In[23]:


metrics.shape


# In[24]:


similarity = cosine_similarity(metrics)


# In[25]:


def recommend(Job_Title):
  index = df[df['Job Title'] == Job_Title].index[0]
  distances = sorted(list(enumerate(similarity[index])),reverse = True , key = lambda x:x[1])
  new_index = [i[0] for i in distances[1:11]]
  return df.iloc[new_index , :-1]


# In[29]:


df.sample(5)


# In[27]:


recommend('Interior Designer')


# In[28]:


recommend('Senior Associate')


# In[30]:


recommend('Solution Director')


# In[ ]:




