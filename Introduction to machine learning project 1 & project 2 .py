#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Introduction to Machine Learning ==> 

# Data ==> 
# (1). Categorical(Descrete) ==> 'yes' , 'no' , 'word'   
# (2). Numerical ==> age(year,month,days) , weight , height 

# Data ==>(1). Input(independent) data (2). Target data 

# Machine Learning Algorithm ==> Depend ==> Target data 

# correlation ==> 

# +1 ==> 1 column increase  , 2 column increase 
# -1 ==> 1 column increase , 2 column decrease 
# 0 ==> 1 column increase/decrease , 2 column not effect .


# In[3]:


# Steps of Apply  Machine Learning on a dataset ==> 
# (1). data ==> x(input column) , y(depdent/target column) 
# (2). data ==> sklearn library ==> train , test 
# Example ==> child ==> fire ==> trainnig data  , child ==> fire ==> no , yes(testing)

# x(input data) ==> x_train , x_test 
# y(target data) ==> y_train , y_test 

# training ==> higher data 
# testing ==> lower 


# In[4]:


# (3). How i select which type of algorithm i have to use ? 
# Algorithm ==> 
# (a). Supervised  ==> Column Label given (data points + column name)
# (b). UnSupervised ==> Column Lbael is not given(data points) 


# In[5]:


# Target data ==> 
# (1). Numerical ==> 
# (a). Linear Regression 
# (b). Decision TreeRegressor 
# (c). RandomForestRegressor 

# (2). Categorical ==> 
# (a). Logistic Regression 
# (b). Decision Tree Classiifier 
# (c). Random Forest Classifier 
# (d). Naive Bayes ==> Input column independet  , Bayes theorm ===> Text Classification 

# UnSupervised data ==> 
# (1). PCA(Principal Component Analysis) ==> Dimensionality Reduction 
# (2).Anomoly Detection(Apriori) ==> Customer(Bread , Milk , Butter) , (Notebook , Pen) etc . 
# (3). Kmeans ==> Cluster ==> App ==> Churn , Continue , leave 


# In[6]:


# Measurement of Algorithm's performance ==> 

# (1).Categorical ==> Accuracy_score(test_data , predicted_data)
# (2). Numerical ==> MSE , MAE , RMSE , r2_score() 


# # Project1(Predicted Salary of an Employee) 

# In[1]:


import numpy as np 
import pandas as pd 


# In[3]:


df = pd.read_csv("C:\\Users\\Dell\\Downloads\\Salary_Data.csv")


# In[4]:


df.head() 


# In[5]:


import plotly.express as px  # Data visualization library 


# In[6]:


import plotly.graph_objects as go 


# In[7]:


df.isnull().sum() 


# In[8]:


import plotly.express as px
figure = px.scatter(data_frame = df , 
                   x = 'Salary',
                   y = 'YearsExperience',
                   size=  'YearsExperience',
                   trendline = 'ols')
figure.show()


# In[9]:


x = np.asanyarray(df[['YearsExperience']]) #Input column 
y = np.asanyarray(df[['Salary']])  #Target column 


# In[10]:


from sklearn.model_selection import train_test_split 


# In[11]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42) 


# In[12]:


x.shape


# In[13]:


print(x_train.shape)
print(x_test.shape)


# In[14]:


from sklearn.linear_model import LinearRegression 


# In[15]:


lr = LinearRegression() 


# In[16]:


lr.fit(x_train , y_train) #fit means learn the parameter 


# In[17]:


a = float(input("Enter your years of experience:"))
b = np.array([[a]])
print("Predicted Salary=" , lr.predict(b)) 


# # Project-2(Employee Attrition Analysis) 
# 
# Employee Attrition analysis is a type of behavioural analysis where we study the behaviour amd characteristivs of the employees who left the organization and compare their charateristics with the current employees to find the employees who may leave the organization soon .
# 
# A high rate of attrition of employees can be expensive for any company in terms of recruitment and training costs , loss of productivity and moracle reduction of employees . By identifying the casue of attrition, a company can take measures to reduce the attrition of employees and maintain precious employees . 

# In[18]:


import numpy as np 
import pandas as pd 


# In[19]:


df=  pd.read_csv("C:\\Users\\Dell\\Downloads\\Attrition.csv")


# In[20]:


df.head() 


# In[21]:


import plotly.graph_objects as go 
import plotly.io as pio 
pio.templates.default = "plotly_white" 


# In[22]:


# df.isnull().sum() 


# In[23]:


# Filter the data to show only 'yes' values in the attrition column. 


# In[24]:


attr_df = df[df['Attrition']=='Yes']
attr_df


# # Calculate the attrition by Department 

# In[25]:


attr_by_dpt = attr_df.groupby(['Department']).size().reset_index(name="count") 


# In[26]:


fig = go.Figure(data = [go.Pie(
    labels = attr_by_dpt['Department'],
    values = attr_by_dpt['count'],
    hole = 0.4 , 
    marker = dict(colors = ['yellow' , 'green']),
    textposition = 'inside'
)])

fig.update_layout(title = 'Attrition by Department')
fig.show() 


# # Calculate  the attrition by EducationField

# In[27]:


attr_by_edu = attr_df.groupby(['EducationField']).size().reset_index(name="count") 


# In[28]:


# attr_by_edu


# In[30]:


fig = go.Figure(data = [go.Pie(
    labels = attr_by_edu['EducationField'],
    values = attr_by_edu['count'],
    hole = 0.4 , 
    marker = dict(colors = ['yellow' , 'green']),
    textposition = 'inside'
)])

fig.update_layout(title = 'Attrition by EducationField')
fig.show() 


# # Now let's have a look at the percentage of attrition by number of years at the company 

# In[31]:


attr_by_comp = attr_df.groupby(['YearsAtCompany']).size().reset_index(name = "count") 
# attr_by_comp


# # Now let's have a look at the percentage of attrition by Permotion . 

# In[32]:


attr_by_promo = attr_df.groupby(['YearsSinceLastPromotion']).size().reset_index(name="count")
attr_by_promo


# # Now let's have a look at the percentage of attrition by gender

# In[33]:


attr_by_gender = attr_df.groupby(['Gender']).size().reset_index(name = "count") 
attr_by_gender


# In[34]:


fig = go.Figure(data = [go.Pie(
    labels = attr_by_gender['Gender'],
    values = attr_by_gender['count'],
    hole = 0.4 , 
    marker = dict(colors = ['yellow' , 'green']),
    textposition = 'inside'
)])

fig.update_layout(title = 'Attrition by Department')
fig.show() 


# # Now let's have a look at the attrition by analyzing the relationship between monthly income and the age of the employees . 

# In[35]:


fig = px.scatter(df , 
                x = 'Age',
                y = 'MonthlyIncome',
                color = 'Attrition',
                trendline = 'ols')
fig.show()


# In[ ]:


# categorical ==> barplot, pie chart , donut chart
# numerical ==> line plot , histogram , distplot 

