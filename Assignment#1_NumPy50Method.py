#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


l=np.arange(1000)


# In[3]:


np.zeros((4,4))


# In[4]:


a=np.ones((4,4))


# In[5]:


np.empty((4,4))


# In[6]:


x=np.random.randn((105))


# In[7]:


np.abs((4))


# In[8]:


np.average(100)


# In[9]:


np.short([12,43,2,6])


# In[10]:


np.sqrt(66)


# In[11]:


np.sum([12,45,45])


# In[12]:


np.subtract(100,25)


# In[13]:


np.amax([12,34,54,35,56])


# In[14]:


np.min([12,34,56,76,23])


# In[15]:


np.minimum(23,45)


# In[16]:


np.mean([23,45,34])


# In[17]:


np.mod(23,34)


# In[18]:


np.median([12,34,21,23])


# In[19]:


np.prod([12,2,2])


# In[20]:


np.product([12,2,3,])


# In[21]:


np.any([12,34,32,-9])


# In[22]:


np.all([12,34,221])


# In[23]:


np.absolute([12,23,43,45])


# In[24]:


np.sort([12,34,54,23,12])


# In[25]:


np.unique([12,34,23,13,12,13])


# In[26]:


l


# In[27]:


np.save("filel",l)


# In[28]:


np.load("filel.npy")


# In[29]:


np.random.normal(4,4)


# In[30]:


np.array(4)


# In[31]:


np.shape((4,4))


# In[32]:


np.dot(4,6)


# In[33]:


np.add(45,5)


# In[34]:


np.append(l,56)


# In[35]:


np.asanyarray(l)


# In[36]:


np.bytes0([1,2,6,8,9])


# In[37]:


np.alltrue(l)


# In[38]:


np.atleast_1d(1,4,8,9)


# In[39]:


np.allclose(x,4)


# In[40]:


np.random.uniform((4,5))


# In[41]:


np.count_nonzero(x)


# In[ ]:





# In[42]:


q=np.array(16)


# In[43]:


np.alen(a)


# In[44]:


q


# In[45]:


w=np.array(17)


# In[46]:


np.intersect1d(q,w).size


# In[47]:


d=np.random.randn(4,5)


# In[48]:


d


# In[49]:


d.reshape((2,2,-1),order="f")


# In[50]:


d.ravel()


# In[51]:


d


# In[52]:


d.flatten()


# In[53]:


d.reshape((2,2,5))


# In[54]:


d.resize()


# In[55]:


d


# In[56]:


x=[[1,3,5,7],[2,4,6,8]]
y=[[11,13,17,19],[12,14,16,18]]


# In[57]:


x1=np.array(x)


# In[58]:


y1=np.array(y)


# In[59]:


np.concatenate((x1,y1),axis=0)


# In[60]:


np.concatenate((x1,y1),axis=1)


# In[61]:


np.vstack((x1,y1))


# In[62]:


np.hstack((y1,x1))


# In[63]:


v=np.array([1,2,3,4,5,6,7,8,9])


# In[64]:


v


# In[65]:


np.split(v,[2,6])


# In[66]:


e=([1,2,3,4,5,6,7,8,9],[11,12,13,14,15,16,17,18,19])


# In[67]:


e


# In[68]:


e1=np.array(e)


# In[69]:


np.split(e1,[3,6],axis=1)


# In[70]:


v


# In[71]:


np.abs(v)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




