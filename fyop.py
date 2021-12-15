#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import packages

import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from kmodes.kmodes import KModes


# In[2]:


#Importing the csv
df = pd.read_csv(r"C:\Users\Rober\Downloads\pickup2000.csv")
for c in df.columns[df.dtypes == object]: # data types = object
    df[c] = df[c].astype('category')


# In[3]:


df.dropna(subset = ["CarMaker","Make","Type","Year","SpecificProblem","Generalproblems"], inplace = True)
#Drop NaN


# In[4]:


#selecting just the "Car Make" and "General Problems"
df = df[["Make","Generalproblems"]]
df.head()


# In[5]:


# Cost as an empty list to start
cost = []
# K-Modes Clustering code chunk for "Elbow Method"
# This will help with chosing an appropriate K-valie 
K = range(1,5)
for num_clusters in list(K):
    kmode = KModes(n_clusters = num_clusters, init = "Cao", n_init = 1, verbose = 1)
    kmode.fit_predict(df)
    cost.append(kmode.cost_)
    
#plotting curve for the "Elbow Method"
plt.plot(K, cost, 'bx-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show()

#From the plot it appears that 3 is a good K-Value


# In[6]:


# K-modes code chunk for k-value of 3
km = KModes(n_clusters=3, init = "Cao", n_init = 1, verbose=1)
cluster_labels = km.fit_predict(df)
df['Cluster'] = cluster_labels


# In[7]:


# Plotting the clusers using sns.countplot
for col in df:
    plt.subplots(figsize = (15,5))
    sns.countplot(x='Cluster',hue=col, data = df)
    plt.show()


# In[ ]:




