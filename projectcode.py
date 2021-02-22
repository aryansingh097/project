#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd


# In[9]:


df1=pd.read_csv('https://api.covid19india.org/csv/latest/raw_data1.csv')
df2=pd.read_csv('https://api.covid19india.org/csv/latest/raw_data2.csv')
df3=pd.read_csv('https://api.covid19india.org/csv/latest/raw_data3.csv')
df4=pd.read_csv('https://api.covid19india.org/csv/latest/raw_data4.csv')
df5=pd.read_csv('https://api.covid19india.org/csv/latest/raw_data5.csv')
df6=pd.read_csv('https://api.covid19india.org/csv/latest/raw_data6.csv')
df7=pd.read_csv('https://api.covid19india.org/csv/latest/raw_data7.csv')
df8=pd.read_csv('https://api.covid19india.org/csv/latest/raw_data8.csv')


# In[10]:


df1.info()


# In[11]:


df1.describe()


# In[12]:


df1.columns


# In[13]:


df3.columns


# In[14]:


df2.columns


# In[15]:


df1=df1.loc[:,['Num Cases','Date Announced','Age Bracket','Gender','Detected City','Detected District', 'Detected State','Current Status']]
df2=df2.loc[:,['Num Cases','Date Announced','Age Bracket','Gender','Detected City','Detected District', 'Detected State','Current Status']]
df3=df3.loc[:,['Num Cases','Date Announced','Age Bracket','Gender','Detected City','Detected District', 'Detected State','Current Status']]
df4=df4.loc[:,['Num Cases','Date Announced','Age Bracket','Gender','Detected City','Detected District', 'Detected State','Current Status']]
df5=df5.loc[:,['Num Cases','Date Announced','Age Bracket','Gender','Detected City','Detected District', 'Detected State','Current Status']]
df6=df6.loc[:,['Num Cases','Date Announced','Age Bracket','Gender','Detected City','Detected District', 'Detected State','Current Status']]
df7=df7.loc[:,['Num Cases','Date Announced','Age Bracket','Gender','Detected City','Detected District', 'Detected State','Current Status']]
df8=df8.loc[:,['Num Cases','Date Announced','Age Bracket','Gender','Detected City','Detected District', 'Detected State','Current Status']]


# In[16]:


df=df1.append([df2,df3,df4,df5,df6,df7,df8])


# In[17]:


df.info()


# In[18]:


Date=df['Date Announced'].str.split('/',expand=True)


# In[19]:


Date.columns=['Day','Month','Year']
Date


# In[20]:


df=pd.concat([df,Date],axis=1)


# In[21]:


df.info()


# In[22]:


df


# In[55]:


df.info()


# In[56]:


df.to_csv('Covid-19.csv')


# In[23]:


data=pd.read_csv('Covid-19.csv')


# In[24]:


data


# In[25]:


#percent missing data
data.isnull().sum(axis=0).sort_values(ascending=False)/len(data)*100


# In[60]:


#total covid-19 cases month wise
m=data[data['Current Status']=='Hospitalized'].groupby('Month')['Num Cases'].sum()


# In[61]:


m.plot.bar()


# In[62]:


m


# In[63]:


#effected male/female
m=data.groupby('Gender')['Num Cases'].sum()


# In[64]:


m


# In[65]:


m.plot.bar()


# In[66]:


# age effected
m=data.groupby('Age Bracket')['Num Cases'].sum().sort_values(ascending=False).head(10)


# In[67]:


m


# In[70]:


m.plot.bar(figsize=(15,5))


# In[75]:


# statewise total cases in India
m=data[data['Current Status']=='Hospitalized'].groupby('Detected State')['Num Cases'].sum().sort_values(ascending=False).head(5)


# In[76]:


m


# In[77]:


m.plot.bar()


# In[26]:


# howmmany cases every day india
m=data[data['Current Status']=='Hospitalized'].groupby(['Month','Day'])[['Num Cases']].sum()


# In[80]:


m


# In[95]:


# separate plot for every month data
m.unstack(level=0).plot(subplots=True)
m.plot.bar()


# In[96]:


data['Current Status'].unique()


# In[97]:


m=data[data['Current Status']=='Deceased']['Num Cases'].sum()


# In[98]:


m


# In[99]:


#statewise dead patient
m=data[data['Current Status']=='Deceased'].groupby('Detected State')['Num Cases'].sum().sort_values(ascending=False)


# In[100]:


m


# In[29]:


Day=data[data['Current Status']=='Hospitalized'].groupby(['Month','Day'])[['Num Cases']].sum()


# In[31]:


Day


# In[32]:


len(Day)


# In[33]:


x=np.arange(len(Day))


# In[34]:


y=Day.values


# In[36]:


import matplotlib.pyplot as plt


# In[35]:


x=x.reshape(-1,1)


# In[46]:


# machine learning prediction
from sklearn.linear_model import LinearRegression


# In[38]:


regressor=LinearRegression()


# In[39]:


regressor.fit(x,y)


# In[40]:


y[121]


# In[51]:


regressor.predict([[152]])


# In[42]:


plt.plot(x,y)


# In[43]:


yp=regressor.predict(x)


# In[50]:


plt.scatter(x,y)
plt.plot(x,yp)
plt.show()


# In[45]:


regressor.score(x,y)*100


# In[ ]:




