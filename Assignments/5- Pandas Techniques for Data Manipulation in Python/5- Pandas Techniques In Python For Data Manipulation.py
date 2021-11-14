#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[11]:


data= pd.read_csv('iris_training.csv')
data


# In[12]:


data.head()


# In[13]:


#Boolean Indexing in Pandas
data.loc[(data["SepalLengthCm"]>=5) & (data["SepalWidthCm"]<=3) & (data["PetalLengthCm"]>1.2), ["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "Species"]]


# In[16]:


#apply function
def missingValues(x):
    return sum(x.isnull())
print("Number of missing elements column wise:")
print(data.apply(missingValues, axis=0))
print("\nNumber of missing elements row wise:")
print(data.apply(missingValues, axis=1).head())


# In[19]:


#RenameColumns
newcols={
"Id":"id",
"SepalLengthCm":"sepallength",
"SepalWidthCm":"sepalwidth"}
 
data.rename(columns=newcols,inplace=True)
 
print(data.head())


# In[20]:


#Slice Data
#data[start:end]
#start is inclusive whereas end is exclusive
print(data[10:21])
# it will print the rows from 10 to 20.
 
# you can also save it in a variable for further use in analysis
sliced_data=data[10:21]
print(sliced_data)


# In[22]:


#Plotting of boxplots and histograms
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data.boxplot(column="sepallength",by="Species")


# In[23]:


data.hist(column="sepallength",by="Species",bins=30)


# In[ ]:




