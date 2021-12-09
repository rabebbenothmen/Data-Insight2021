#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import Libraries
import numpy as np
import pandas as pd


# In[4]:



book = pd.read_csv('BL-Flickr-Images-Book.csv')
book.head()


# In[5]:


#Dropping Columns in a DataFrame
to_drop = ['Edition Statement',
            'Corporate Author',
            'Corporate Contributors',
            'Former owner',
            'Engraver',
            'Contributors',
            'Issuance type',
            'Shelfmarks']
book.drop(to_drop, inplace=True, axis=1)


# In[6]:


book.head()


# In[7]:


#Changing the Index of a DataSet
book['Identifier'].is_unique


# In[8]:


book = book.set_index('Identifier')
book.head()


# In[16]:


book.loc[206]


# In[20]:


#Using .str methods to clean columns
book['Place of Publication'].head(10)


# In[21]:


book.loc[4157862]


# In[23]:


book.loc[4159587]


# In[24]:


pub = book['Place of Publication']
book['Place of Publication'] = np.where(pub.str.contains('London'), 'London',
    np.where(pub.str.contains('Oxford'), 'Oxford',
        np.where(pub.eq('Newcastle upon Tyne'),
            'Newcastle-upon-Tyne', book['Place of Publication'])))


book.head()


# In[10]:


#Cleaning columns using the .apply function
unwanted_characters = ['[', ',', '-']

def clean_dates(item):
    dop= str(item.loc['Date of Publication'])
    
    if dop == 'nan' or dop[0] == '[':
        return np.NaN
    
    for character in unwanted_characters:
        if character in dop:
            character_index = dop.find(character)
            dop = dop[:character_index]
    
    return dop

book['Date of Publication'] = book.apply(clean_dates, axis = 1)


# In[11]:


book.head()


# In[25]:


university_towns = []

with open('university_towns.txt', 'r') as file:
    items = file.readlines()
    states = list(filter(lambda x: '[edit]' in x, items))
    
    for index, state in enumerate(states):
        start = items.index(state) + 1
        if index == 49: #since 50 states
            end = len(items)
        else:
            end = items.index(states[index + 1])
            
        pairs = map(lambda x: [state, x], items[start:end])
        university_towns.extend(pairs)
        
towns_df = pd.DataFrame(university_towns, columns = ['State', 'RegionName'])
towns_df.head()


# In[26]:


olympics_df = pd.read_csv('olympics.csv')
olympics_df.head()


# In[27]:


olympics_df = pd.read_csv('olympics.csv', skiprows = 1, header = 0)
olympics_df.head()


# In[30]:


new_names =  {'Unnamed: 0': 'Country',
              '? Summer': 'Summer Olympics',
              '01 !': 'Gold',
              '02 !': 'Silver',
              '03 !': 'Bronze',
              '? Winter': 'Winter Olympics',
              '01 !.1': 'Gold.1',
              '02 !.1': 'Silver.1',
              '03 !.1': 'Bronze.1',
              '? Games': '# Games', 
              '01 !.2': 'Gold.2',
              '02 !.2': 'Silver.2',
              '03 !.2': 'Bronze.2'}

olympics_df.rename(columns = new_names, inplace = True)
olympics_df.head()


# In[34]:


#visualizing data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
iris = pd.read_csv("Iris.csv")
iris.head()


# In[38]:


iris["Species"].value_counts()


# In[39]:


iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")


# In[40]:


sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)


# In[41]:


sns.FacetGrid(iris, hue="Species", size=5)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()


# In[42]:


sns.boxplot(x="Species", y="PetalLengthCm", data=iris)


# In[43]:


sns.FacetGrid(iris, hue="Species", size=6)    .map(sns.kdeplot, "PetalLengthCm")    .add_legend()


# In[ ]:




