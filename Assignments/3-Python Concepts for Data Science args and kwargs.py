#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Python *args

# packing explicite
def area_rectangle(*args):  # the arguments passed in parameter are packaged in args which behaves like a tuple 
    if len(args) == 2:
        return args[0]*args[1]
    else:
        print('Please state two parameters')

#Python *kwargs
def area_rectangle2(**kwargs):  # the arguments passed in parameter are packaged in kwargs which behaves like a dictionary 
    if len(kwargs) == 2:
        result = 1
        for key, value in kwargs.items():
            result *=value
        return result
    else:
        print('Please state two parameters')

if __name__ == '__main__':
    # A list will be created from the arguments provided 
    print (area_rectangle(3,8))
    # A dictionary will be created from the named arguments
    print (area_rectangle2(cote1=4, cote2=8))


# In[ ]:




