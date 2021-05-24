#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data =  pd.read_csv('train_set.csv')
data.tail()


# In[2]:


data.isnull().any()


# In[3]:


print(data.shape)
data = data.dropna()
print(data.shape)


# In[4]:


data1 = pd.get_dummies(data,drop_first=True)
print(data1.shape)
#due to high computational power requirment i have taken only 100000
data2 = data1.iloc[:10000,:]


# In[5]:


x=data2.drop('Total_Compensation',axis = 1)
y = data2['Total_Compensation']


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# In[7]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 3)
model = KNeighborsRegressor(n_neighbors=3)
model.fit(x_train,y_train)


# In[8]:


y_pred = model.predict(x_test)
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))


# # just for experiment

# In[9]:


data3 = data1.iloc[0:1000,:]
x=data3.drop('Total_Compensation',axis = 1)
y = data3['Total_Compensation']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 3)


# In[10]:


lst1 = []
lst2 = []
for i in range(1,20):
    model = KNeighborsRegressor(n_neighbors=i)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(model.score(x_train,y_train),model.score(x_test,y_test))
    lst1.append(model.score(x_train,y_train))
    lst2.append(model.score(x_test,y_test))


# In[ ]:




