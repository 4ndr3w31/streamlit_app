#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[4]:


iris = load_iris()
X, y = iris.data, iris.target


# In[5]:


X_train,X_test,y_train,y_test = train_test_split (X,y,test_size=0.2,random_state = 42)


# In[6]:


rf_model = RandomForestClassifier(n_estimators = 10,random_state = 42)
rf_model.fit(X_train,y_train)


# In[7]:


y_pred = rf_model.predict(X_test)


# In[10]:


accuracy = accuracy_score(y_test,y_pred)
print(f"Model Accuracy ; {accuracy:.2f}")


# In[11]:


import pickle as pkl
filename = "ModelDep.pkl"
pkl.dump(rf_model,open(filename,'wb'))


# In[ ]:




