#!/usr/bin/env python
# coding: utf-8

# # EDA(Exploratory Data Analysis)

# # Parts of EDA 
# 
# (1). Univariate Analysis ==> Analysis on a sinlge Independent Column 
# 
# (2). Bivariate Analysis ==> Analysis on two columns 
# 
# (3). Multivariate Analysis  ==> Analysis on more than 2 columns 

# # Data Types ==> 
# 
# # (1). Descrete or Categorical Data ==> Marital Status , Total Number of Employees , Gender ....etc 
# 
# # (2). Continuous or Numerical Data ==> Age(Year , month , days) , Height , Weight ...etc 

# In[2]:


import numpy as np   #Scientific Calculation library (open -source library)
import pandas as pd  #Data Manipulations(open - source library) 


# In[4]:


import matplotlib.pyplot as plt  #Visualization library
import seaborn as sns   # matplotlib's upgraded version is seaborn 


# In[5]:


df = pd.read_csv("C:\\Users\\saurabh\\Desktop\\Newdat\\titanic.csv")


# In[6]:


df.head() 


# In[7]:


df.info() 


# # 1 . Univariate Analysis

# In[9]:


df['Survived'].value_counts()   # it returns total count of each sub-category . 


# In[10]:


sns.countplot(x = df['Survived']) 


# In[12]:


df['Survived'].value_counts().plot(kind = 'bar') 


# In[14]:


df['Pclass'].value_counts() 


# In[15]:


sns.countplot(x = df['Pclass']) 


# # If we want to find out percentage then use piechart 

# In[17]:


df['Survived'].value_counts()  


# In[21]:


df['Survived'].value_counts().plot(kind = 'pie' , autopct = '%.3f' )


# In[28]:


df['Pclass'].value_counts().plot(kind = 'pie' , autopct = '%.f' )


# In[30]:


# If we have Numerical data them we can use Histogram because it finds distribution . 


# In[32]:


plt.hist(x = df['Age'])
plt.show()


# In[33]:


# Distplot(Numerical data) 
# curve ==> KDE(Kernal Density Extraction) 
# use for find Probability . 


# In[34]:


sns.distplot(x = df['Age']) 


# In[35]:


sns.distplot(x = df['Age'] , hist = False) 


# In[36]:


# Boxplot ==> For find our outliers(data ponts ==> mean update) . 

# 1 . Lower Fence 
# 2 . 25% of data 
# 3 . IQR(Inter Quartile Range) 
# 4. 75%  data 
# 5. Upper Fence 


# In[37]:


sns.boxplot(x = df['Age']) 


# In[38]:


df = pd.read_csv("C:\\Users\\saurabh\\Desktop\\Newdat\\tips.csv")


# In[39]:


df.head()


# # Bivariate Analysis 

# # 1 . scatterplot(Numerical column - Numerical column) 

# In[40]:


sns.scatterplot(x = df['total_bill'],
               y = df['tip']) 


# In[41]:


sns.scatterplot(x = df['total_bill'],
               y = df['tip'],
               hue = df['sex']) 


# In[42]:


sns.scatterplot(x = "total_bill", 
               y = "tip", 
                data = df , 
                hue = df['sex'], 
                style = df['smoker'] 
               )


# In[43]:


df = pd.read_csv("C:\\Users\\saurabh\\Desktop\\Newdat\\titanic.csv")


# In[44]:


df.head() 


# In[45]:


sns.barplot(x = df['Pclass'],
           y = df['Age'])


# In[47]:


sns.barplot(x = df['Pclass'],
           y = df['Age'] , 
           hue = df['Sex'])


# In[48]:


sns.distplot(df[df['Survived'] == 0]['Age'] , hist = False) 


# # HeatMap

# In[51]:


p = pd.crosstab(df['Pclass'],df['Survived'])
p


# In[52]:


sns.heatmap(p)


# In[55]:


((df.groupby('Pclass').mean()['Survived'])*100).plot(kind= 'bar')


# In[56]:


df.groupby('Pclass').mean()


# # How to upload data into GitHub 

# In[57]:


# Github ==>platform ==> multiplr users ==> each-other(connect) 

# 1 project ==> 5 members (complete)  ==> work from home ==> 1 user (code push , pull) ==> 4 users (files update) ==> github


# In[ ]:


# github ==> create account 
# git bash download 


# In[58]:


# step-1  Create a New Folder on Desktop using mkdir 


# In[59]:


# step-2  After then we have to need switch our directories using cd command and jump on this folder . 


# In[60]:


# cd folderName/ 


# In[61]:


# step-3 Now we can run git init command for initialize our folder form git 
# git init 


# In[62]:


# step-4 Now we will create some new files in folderName using touch command . 
# touch abc.py 
# touch pqr.html 


# In[63]:


# step-5 Nw we can check current status of data  
# git status 


# In[64]:


# step-6 We will add or  files using git add . 

# git add . 


# In[65]:


# step-7 for save our data will use this command . 

# git commit -m "added" 


# In[66]:


# step-7Now we will create our account on github and open github 

# Go on github ==> click on "new" 


# In[68]:


# step-8  Repo Name ==> Local system folder Name  ==> clcik on "Create Repository"


# In[69]:


# Project link ==> github link 


# # If you have already data in your folder then 

# In[70]:


#(1).  open git bash terminal on that folder . 
# (2). git init 
# (3). git status 
# (4). git add . 
# (5). git commit -m "added" 

# (6). go on github ==> create new repository ==> last 2 lines of code copy ==> git bash (paste) 


# In[ ]:




