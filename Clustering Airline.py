#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[11]:


airline= pd.read_csv("C:/Users/User/Desktop/EastWestAirlines1.CSV")


# In[12]:


airline


# In[13]:


airline.info()


# In[14]:


airline2=airline.drop(['ID#'],axis=1)
airline2


# In[15]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[16]:


airline2_norm = norm_func(airline2)
airline2_norm


# In[ ]:


plt.figure(figsize=(10, 7))  
dendograms=sch.dendrogram(sch.linkage(airline2_norm,'complete'))


# In[ ]:


hclusters=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
hclusters


# In[ ]:


y=pd.DataFrame(hclusters.fit_predict(airline2_norm),columns=['clustersid'])
y['clustersid'].value_counts()


# In[ ]:


airline2['clustersid']=hclusters.labels_
airline2


# In[ ]:


airline2.groupby('clustersid').agg(['mean']).reset_index()


# In[ ]:


plt.figure(figsize=(10, 7))  
plt.scatter(airline2['clustersid'],airline2['Balance'], c=hclusters.labels_)


# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


# In[ ]:


airline2_norm


# In[ ]:


model = KMeans(n_clusters=5)
model.fit(airline2_norm)
model.labels_


# In[ ]:


md = pd.Series(model.labels_)
airline2['clusters'] = md
airline2


# In[ ]:


airline2.groupby(airline2.clusters).mean()


# In[ ]:


plt.figure(figsize=(10, 7))  
plt.scatter(airline2['clusters'],airline2['Balance'], c=model.labels_) 


# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


# In[ ]:


airline2_norm


# In[ ]:


dbscan = DBSCAN(eps=0.8, min_samples=15)
dbscan.fit(airline2_norm)


# In[ ]:


dbscan.labels_


# In[ ]:


ml=pd.DataFrame(dbscan.labels_,columns=['cluster'])
pd.concat([airline2_norm,ml],axis=1)


# In[ ]:




