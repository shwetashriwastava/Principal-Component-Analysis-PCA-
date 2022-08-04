#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn


# In[4]:


wine = pd.read_csv("D:\\data science\\assignments\\ass-8 PCA\\wine.csv")
wine


# # Hierarchical Analysis

# In[5]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[7]:


winenorm = norm_func(wine.iloc[:,1:])
winenorm


# In[8]:


dendrogram = sch.dendrogram(sch.linkage(winenorm, method='single'))


# In[9]:


hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'single')


# In[10]:


y_hc = hc.fit_predict(winenorm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[12]:


Clusters


# In[13]:


dendrogram = sch.dendrogram(sch.linkage(winenorm, method='complete'))


# In[14]:


dendrogram = sch.dendrogram(sch.linkage(winenorm, method='average'))


# In[15]:


dendrogram = sch.dendrogram(sch.linkage(winenorm, method='centroid'))


# In[16]:


hc  = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'complete')
hc1 = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'single')
hc2 = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'centroid')
hc3 = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'average')


# In[18]:


y_hc = hc.fit_predict(winenorm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])

y_hc1 = hc.fit_predict(winenorm)
Clusters_1 =pd.DataFrame(y_hc,columns=['Clusters1'])

y_hc2 = hc.fit_predict(winenorm)
Clusters_2 =pd.DataFrame(y_hc,columns=['Clusters2'])


y_hc3 = hc.fit_predict(winenorm)
Clusters_3 =pd.DataFrame(y_hc,columns=['Clusters3'])


# In[19]:


Clusters


# In[20]:


Clusters_1


# In[21]:


Clusters_2


# In[22]:


Clusters_3


# # K mean

# In[24]:


import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
import numpy as np


# In[25]:


wine = pd.read_csv("D:\\data science\\assignments\\ass-8 PCA\\wine.csv")
wine


# In[26]:


X = np.random.uniform(0,1,1000)
Y = np.random.uniform(0,1,1000)
xy =pd.DataFrame(columns=["X","Y"])
xy.X = X
xy.Y = Y
xy.plot(x="X",y = "Y",kind="scatter")


# In[27]:


X = np.random.uniform(0,1,1000)
X


# In[28]:


model1 = KMeans(n_clusters=10).fit(xy)

xy.plot(x="X",y = "Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)


# In[32]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df_norm = norm_func(wine.iloc[:,1:])
df_norm.head(5)


# In[33]:


model=KMeans(n_clusters=5) 
model.fit(df_norm)

model.labels_


# In[36]:


md=pd.Series(model.labels_)    
wine['clust']=md              
df_norm.head()

md=pd.Series(model.labels_)
wine['clust']=md
df_norm.head()


# In[37]:


wine.iloc[:,1:7].groupby(wine.clust).mean()


# In[38]:


wine.head()


# In[ ]:





# In[ ]:





# In[ ]:




