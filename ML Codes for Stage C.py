#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Data_for_UCI_named.csv')
data.head()


# In[14]:


#Checking the distribution of the Label (stabf)
data[ 'stabf' ].value_counts()

#converting categorical variable in column stabf to numerical
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
data.stabf = LE.fit_transform(data.stabf)
data.head()

x = data.drop(columns= 'stabf' )
y = data['stabf'] 

#Splitting data into training and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#Using standard scaler to transform data
from sklearn.preprocessing import StandardScaler
S_scaler = StandardScaler()
scaled_data = S_scaler.fit_transform(data)

#Training the model using Random forest and extra trees classifier
#Importing performance evaluation models
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

RFC = RandomForestClassifier(random_state=1)
RFC.fit(x_train, y_train)
RandomForestClassifier(random_state=1)
y_pred = RFC.predict(x_test)
accuracy_score(y_pred, y_test)

#Importing LGBM
import lightgbm as LGBM


# In[ ]:




