#!/usr/bin/env python
# coding: utf-8

# # Project-3(Social Media Ads Click Through Rate Prediction)

# In[2]:


import numpy as np 
import pandas as pd 


# In[3]:


df = pd.read_csv("C:\\Users\\Dell\\Documents\\data analyst task\\click.csv")
df


# In[4]:


df.head() 


# In[5]:


import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio 
pio.templates.default = "plotly_white" 


# # The "Clicked on Ad" column contains 0 and 1 values , where 0 means not clicked and 1 means clicked . Now we will transform these values into 'yes' and 'no' . 

# In[6]:


df['Clicked on Ad'] = df['Clicked on Ad'].map({0:'No' , 1:'Yes'})


# In[7]:


# df['Clicked on Ad']


# # Now let's analyze the click throgh rate based on the time spent by the users on the website 

# In[8]:


fig = px.box(df,
    x = "Daily Time Spent on Site",
    color = "Clicked on Ad",
    title = "Click Through Rate based Time Stamp on Site",
    color_discrete_map = {'Yes':'Blue',
                         'No':'Black'}
)
fig.show() 


# In[9]:


# Box Plot ==> Outlier ==> data points ==> meann==> performance update 
# Box plot 
# (1). Lower Fence
# (2). Upper fence 
# (3). 25% of data 
# (4). 75% of data 
# (5). IQR(Inter Quartile Range) ==> (75% - 25%) data 


# # Now let's Analyze the click through rate based on the daily internet usage of the user

# In[10]:


fig = px.box(df,
            x = "Daily Internet Usage",
            color = "Clicked on Ad",
            title = "Click Through Rate Based on the Daily Internet Usage",
            color_discrete_map = {'Yes':'Blue', 'No':'Red'})
fig.show() 


# # Now let's analyze the Click through Rate based on the Age of the users 

# In[11]:


fig = px.box(df,
            x = "Age",
            color = "Clicked on Ad",
            title = "Click Through Rate Based on Age",
            color_discrete_map = {'Yes':'black', 'No':'Red'})
fig.show() 


# # Now let's analyze the click through rate based on the income of the users

# In[12]:


fig = px.box(df , 
            x = "Area Income",
            color = "Clicked on Ad",
            title = "Click Through Rate Based on the Income",
            color_discrete_map = {'Yes':'Blue', 'No':'Red'})
fig.show() 


# # Calculating Click Through Rate of Ads

# In[13]:


df['Clicked on Ad'].value_counts() 


# In[14]:


df.shape


# In[15]:


c_t_r_yes = 4917/10000 *100
c_t_r_yes


# In[16]:


c_t_r_no = 5083/10000 *100
c_t_r_no


# # Click Through Rate Prediction Model 

# In[17]:


df.head(3)


# In[18]:


df['Gender'] = df['Gender'].map({'Male':1 , 'Female':0})


# In[19]:


x = df.iloc[: , 0:7]
x = x.drop(columns = ['Ad Topic Line', 'City'], axis=1)  #Input data 


# In[20]:


y = df.iloc[:,9]   #Target data 


# In[21]:


x.head()


# In[22]:


from sklearn.model_selection import train_test_split # for divide data 


# In[23]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42) 


# In[24]:


x.shape


# In[25]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[26]:


from sklearn.ensemble import RandomForestClassifier 


# In[27]:


rf = RandomForestClassifier()   #make a object of Algorithm 


# In[28]:


rf.fit(x,y) 


# # Now let's test the model by making prediction 

# In[29]:


print("Ads Click Through Rate Prediction:")
a = float(input("Daily Time Spent on Site:"))
b = float(input("Age:"))
c = float(input("Area Income:"))
d = float(input("Daily Internet Usage:"))
e = input("Gender (Male = 1 ,Female = 0):")

features = np.array([[a,b,c,d,e]])
print("Will the User Click on Ad=" , rf.predict(features))


# In[30]:


x.head(2)


# # Project -4 (Number of Orders Prediction)

# In[12]:


import numpy as np 
import pandas as pd 


# In[14]:


df = pd.read_csv("C:\\Users\Dell\\Documents\\data analyst task\\supliment.csv")
df


# In[15]:


df.head()


# In[16]:


df.info()


# In[17]:


df.describe()


# # Now let's  explore some important features from this data to know about the fectors affecting the number of orders for Suppliments

# In[18]:


import plotly.express as px 


# In[19]:


pie = df['Store_Type'].value_counts()
store = pie.index
orders = pie.values 

fig = px.pie(df , values = orders , names = store)
fig.show()


# # Now let's have a look at the distribution of the number of orders , according to the location 

# In[20]:


pie2 = df['Location_Type'].value_counts()
location = pie2.index
orders = pie2.values 

fig = px.pie(df , values = orders , names = location)
fig.show()


# In[21]:


pie3 = df['Discount'].value_counts()
discount = pie3.index
orders = pie3.values 

fig = px.pie(df , values = orders , names = discount)
fig.show()


# In[22]:


df.head(2)


# In[23]:


df['Discount'] = df['Discount'].map({'No':0 , 'Yes':1})
df['Store_Type'] = df['Store_Type'].map({'S1':1 , 'S2':2 , 'S3':3 , 'S4':4})
df['Location_Type'] = df['Location_Type'].map({'L1':1 , 'L2':2, 'L3':3 , 'L4':4 , 'L5':5})


# In[24]:


df.dropna() 


# In[25]:


x = np.array(df[['Store_Type' , 'Location_Type', 'Holiday', 'Discount']])  #Input data 
x


# In[26]:


y = np.array(df['#Order'])  #Target data


# In[27]:


from sklearn.model_selection import train_test_split 


# In[28]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)


# In[29]:


from sklearn.ensemble import RandomForestRegressor 


# In[30]:


rf = RandomForestRegressor()


# In[31]:


rf.fit(x_train , y_train) 


# In[32]:


a = int(input("Enter store type:"))
b = int(input("Enter Location :"))
c = int(input("Enter Holiday:"))
d = int(input("Enter Discount:"))

features = np.array([[a,b,c,d]])
print("Number of Orders = " , rf.predict(features))


# In[ ]:




