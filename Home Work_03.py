#!/usr/bin/env python
# coding: utf-8

# Data from https://www.kaggle.com/blastchar/telco-customer-churn

# In[88]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[89]:


df = pd.read_csv("cars.csv")


# In[90]:


len(df)


# ## Initial data preparation

# In[91]:


df.head()


# In[92]:


df.head().T


# In[93]:


df.columns


# In[94]:


Features_Cat=['Make','Model','Year','Engine HP','Engine Cylinders','Transmission Type','Vehicle Style','highway MPG','city mpg','MSRP']


# In[95]:


data=df[Features_Cat]


# In[96]:


data.head()


# In[97]:


data.columns = data.columns.str.replace(' ', '_').str.lower()


# In[98]:


data.head()


# In[99]:


data= data.fillna(0)


# In[100]:


data.isnull().sum()


# ## Question 1

# In[101]:


data['transmission_type'].mode()


# In[102]:


data.dtypes


# In[103]:


#numerical_columns = data.select_dtypes(include=['int64', 'float64'])
#categorical_columns = data.select_dtypes(include=['object'])


# In[104]:


numerical=['year','engine_hp','highway_mpg','city_mpg']


# In[105]:


categorical = ['make', 'model','transmission_type','vehicle_style']


# # Question 2

# In[106]:


correlation_matrix = data[numerical].corr()


# In[107]:


correlation_matrix 


# In[108]:


data.rename(columns={'msrp': 'Price'}, inplace=True)


# In[109]:


Price_mean=data['Price'].mean()
Price_mean


# In[110]:


data['above_average'] = (data['Price'] > Price_mean).astype(int)


# ## Split the DATA
# 

# In[111]:


from sklearn.model_selection import train_test_split


# In[112]:


df_train_full, df_test = train_test_split(data, test_size=0.2, random_state=42)


# In[113]:


df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)


# In[114]:


len(data),len(df_train),len(df_val),len(df_test)


# In[115]:


df_train.shape


# In[116]:


y_train = df_train.above_average.values



# In[117]:


y_val = df_val.above_average.values


# In[118]:


df_train.columns


# In[119]:


df_train = df_train.drop(columns=['Price','above_average'])


# In[120]:


df_val = df_val.drop(columns=['Price','above_average'])


# ## Question 3

# In[121]:


from sklearn.metrics import mutual_info_score


# In[122]:


def calculate_mi(series):
    return mutual_info_score(series,y_train)


# In[123]:


df_mi = df_train[categorical].apply(calculate_mi)


# In[124]:


df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')


# In[125]:


df_mi['MI'] = df_mi['MI'].round(2)


# In[126]:


display(df_mi.head())
display(df_mi.tail())


# In[ ]:





# ## Question 4

# In[127]:


from sklearn.feature_extraction import DictVectorizer


# In[128]:


train_dict = df_train[categorical + numerical].to_dict(orient='records')


# In[129]:


train_dict[0]


# In[130]:


dv = DictVectorizer(sparse=False)


# In[131]:


dv.fit(train_dict)


# In[132]:


X_train = dv.transform(train_dict)
X_train


# In[133]:


X_train.shape


# In[134]:


dv.get_feature_names_out()


# Training logistic regression

# In[135]:


from sklearn.linear_model import LogisticRegression


# In[136]:


model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)


# In[137]:


model.fit(X_train, y_train)


# In[138]:


val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)


# In[139]:


model.predict_proba(X_val)


# In[140]:


model.fit(X_train, y_train)


# In[141]:


y_pred = model.predict_proba(X_val)[:, 1]


# In[142]:


threshold = 0.5
y_pred_int = (y_pred >= threshold).astype(int)
y_pred_int


# In[143]:


y_val


# In[144]:


accuracy = np.round(accuracy_score(y_pred_int,y_val),2)
print(accuracy)


# ## Question 5

# In[145]:


features = numerical + categorical
features


# In[146]:


orig_score = accuracy


# In[147]:


for c in features:
    subset = features.copy()
    subset.remove(c)
    
    train_dict = df_train[subset].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    dv.fit(train_dict)

    X_train = dv.transform(train_dict)

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    val_dict = df_val[subset].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    y_pred = model.predict(X_val)

    score = accuracy_score(y_val, y_pred)
    print(c, orig_score - score, score)


# ## Question 6

# In[148]:


data = data.drop(columns=['above_average'])


# In[149]:


data['Price']=np.log1p(data['Price'])


# In[150]:


data


# In[151]:


#@ SPLITTING THE DATASET:
df_train_full, df_test = train_test_split(data, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)


# In[152]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[153]:


#@ PREPARING THE DATASET:
y_train = df_train.Price.values
y_val = df_val.Price.values
y_test = df_test.Price.values


# In[154]:


#@ DELETING DATASET:
del df_train['Price']
del df_val['Price']
del df_test['Price']


# Ridge Regression

# In[155]:


train_dict = df_train[categorical + numerical].to_dict(orient='records')


# In[156]:


#@ VECTORIZING THE DATASET:
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)


# In[157]:


from sklearn.metrics import mean_squared_error

for a in [0, 0.01, 0.1, 1, 10]:
    model = Ridge(alpha=a, solver="sag", random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    score = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(a, round(score, 3))

