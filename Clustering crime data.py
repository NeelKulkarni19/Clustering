#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[2]:


crime=pd.read_csv("F:/Dataset/crime_data.csv")


# In[3]:


crime.head()


# In[4]:


crime.drop(["Unnamed: 0"],inplace=True,axis=1)


# In[6]:


crime.head()


# In[7]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[8]:


df_norm = norm_func(crime.iloc[:,:])


# In[9]:


plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
dendrogram = sch.dendrogram(sch.linkage(df_norm, method='complete'))


# In[10]:


agglo= AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'single').fit(df_norm)
agglo


# In[11]:


y_agglo=agglo.fit_predict(df_norm)
clusters=pd.DataFrame(y_agglo,columns=['clusters'])


# In[12]:


clusters


# In[13]:


data = []
for i in range (1,11):
    kmeans= KMeans(n_clusters=i,random_state=0)
    kmeans.fit(df_norm)
    data.append(kmeans.inertia_)
plt.plot(range(1, 11), data)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Data')
plt.show()


# In[14]:


X = np.random.uniform(0,1,1000)
X


# In[15]:


model=KMeans(n_clusters=4) 
model.fit(df_norm) #fit the data which is normalized

model.labels_


# In[16]:


md=pd.Series(model.labels_) 
crime['clust']=md
df_norm.head()


# In[17]:


crime.iloc[:,:].groupby(crime.clust).mean()


# In[18]:


crime.head()


# In[19]:


print(crime.info())


# In[20]:


array=crime.values
array


# In[21]:


st_scaler= StandardScaler().fit(array)
X =st_scaler.transform(array)
X


# In[22]:


dbscan = DBSCAN(eps=0.8, min_samples=5) 
dbscan.fit(X)


# In[23]:


dbscan.labels_


# In[24]:


clt=pd.DataFrame(dbscan.labels_,columns=['cluster'])
clt


# In[25]:


pd.concat([crime,clt],axis=1)


# In[ ]:




