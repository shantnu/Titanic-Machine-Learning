#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd

#The Machine learning alogorithm
from sklearn.ensemble import RandomForestClassifier

# Test train split
from sklearn.model_selection import train_test_split

# Just to switch off pandas warning
pd.options.mode.chained_assignment = None

# Used to write our model to a file
import joblib


# In[23]:


data = pd.read_csv("titanic_train.csv")
data.head()


# In[24]:


data.columns


# In[25]:


median_age = data['age'].median()
print("Median age is {}".format(median_age))


# In[26]:


data['age'].fillna(median_age, inplace = True)
data['age'].head()


# In[27]:


data_inputs = data[["pclass", "age", "sex"]]
data_inputs.head()


# In[28]:


expected_output = data[["survived"]]
expected_output.head()


# In[29]:


data_inputs["pclass"].replace("3rd", 3, inplace = True)
data_inputs["pclass"].replace("2nd", 2, inplace = True)
data_inputs["pclass"].replace("1st", 1, inplace = True)
data_inputs.head()


# In[30]:


data_inputs["sex"] = np.where(data_inputs["sex"] == "female", 0, 1)
data_inputs.head()


# In[31]:


inputs_train, inputs_test, expected_output_train, expected_output_test   = train_test_split (data_inputs, expected_output, test_size = 0.33, random_state = 42)

print(inputs_train.head())
print(expected_output_train.head())


# In[32]:


rf = RandomForestClassifier (n_estimators=100)


# In[33]:


rf.fit(inputs_train, expected_output_train)


# In[34]:


accuracy = rf.score(inputs_test, expected_output_test)
print("Accuracy = {}%".format(accuracy * 100))
assert(accuracy > .77)

# In[35]:


joblib.dump(rf, "titanic_model1", compress=9)


# In[ ]:
