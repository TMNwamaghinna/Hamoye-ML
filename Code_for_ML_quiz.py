#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[25]:


#importing python libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('energydata_complete.csv')
data.head()
data.columns = ('Date', 'Appliances', 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7',                'RH_7', 'T8', 'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2')
data.shape
data.describe()

#determining the R squared value of T2 and T6
F_cols = ['T2']
x = data[F_cols]
y = data.T6

#importing the linear regression module
from sklearn.linear_model import LinearRegression

#Fitting the data into the linear regression model
Lr = LinearRegression().fit(x, y)

#Finding the Rsquare value
Lr.score(x, y)

#importing the mean absolute error module from sklearn
from sklearn.metrics import mean_absolute_error

mean_absolute_error(x, y)

#Importing the square root from the math library
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score

print(sqrt(mean_squared_error(x, y)))
print(Lr.coef_)


# In[ ]:





# In[ ]:




