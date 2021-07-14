#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data =  pd.read_csv('HR_comma_sep.csv.txt')
data.head()


# In[2]:


data.isnull().values.any()


# In[3]:


from sklearn.preprocessing import LabelEncoder


# In[4]:


le = LabelEncoder()
data['salary']= le.fit_transform(data['salary'])


# In[5]:


data1 = pd.get_dummies(data,drop_first= True)
data1


# In[6]:


print(data.shape)
print(data1.shape)


# In[7]:


X = data1.drop('left',axis = 1)#all other features
y = data1.iloc[:,6]#left feature


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)



from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_y_predict = lr_model.predict(X_test)

print(lr_model.score(X_train, y_train))
print(lr_model.score(X_test, y_test))



from sklearn.neighbors import KNeighborsClassifier
knn_class = KNeighborsClassifier(n_neighbors = 3)

knn_class.fit(X_train, y_train)
knn_class_y_predict = knn_class.predict(X_test)

print(knn_class.score(X_train, y_train))
print(knn_class.score(X_test, y_test))


# In[12]:


from sklearn.model_selection import KFold,cross_val_score


# In[14]:


kfold = KFold(n_splits = 4,random_state = 1)
result = cross_val_score(knn_class,X_train,y_train,cv=kfold,scoring ='accuracy')
print(result)




