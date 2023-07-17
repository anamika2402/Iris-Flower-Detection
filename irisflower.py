#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.datasets import load_iris
data = load_iris()


# In[3]:


data


# In[4]:


data.target_names


# In[5]:


df = pd.DataFrame(data.data)
df.head()


# In[6]:


df.columns = data.feature_names
df.head()


# In[7]:


df['Species'] = data.target
df.head()


# In[8]:


X = df.drop('Species',axis=1)
y = df['Species']
print("X Shape:",X.shape)
print("y Shape:",y.shape)


# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=43)


# In[10]:


from sklearn.neighbors import KNeighborsClassifier


# In[11]:


from sklearn.metrics import accuracy_score
k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))


# In[12]:


plt.xlabel("Value of K in KNN")
plt.ylabel("Testing Accuracy")
plt.plot(k_range,scores)


# In[13]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)


# In[14]:


y_pred = knn.predict(X_test)
print("Accuracy Score:{:.2f}%".format(accuracy_score(y_test,y_pred)*100))


# In[15]:


from sklearn.metrics import f1_score


# In[16]:


print("f1 score:",f1_score(y_test,y_pred,average="weighted"))


# In[17]:


df = df.replace({0:"setosa",1:'versicolor',2:'virginica'})


# In[18]:


test = df.sample(1).values
test


# In[19]:


if knn.predict(test[:,0:4]) == 0:
    print("Species : setosa")
elif knn.predict(test[:,0:4]) == 1:
    print("Species : versicolor")
else:
    print("Species : virginica")


# In[ ]:





# In[ ]:




