#!/usr/bin/env python
# coding: utf-8
Syntax Error
# In[5]:


# initialize exam score
x = int(input("What is your exam score?"))
  
# check  whether the student has passed their exam or not 
if(x>=10)
    print("Congrats! you succeed ")
else print("You did not pass the exam")

logical errors(Exception)
# In[6]:


# initialize the amount variable
marks = 10000
  
# perform division with 0
a = marks / 0
print(a)


# In[7]:


a = [1, 2, 3] 
print (a[3]) 


# In[8]:


assert False, 'The assertion failed'


# In[9]:


class Attributes(object):
    pass
  
object = Attributes()
print (object.attribute)


# In[10]:


import module


# In[11]:


array = { 'a':1, 'b':2 }
print (array['c'])


# In[13]:


def func():
    print (ans)
  
func()


# In[21]:


arr = ('tuple', ) + 'string'
print (arr)


# In[26]:


print (int('a'))

Handling Error (try, except, finally)
# In[27]:


# put unsafe operation in try block
try:
     print("code start")
          
     # unsafe operation perform
     print(1 / 0)
  
# if error occur the it goes in except block
except:
     print("an error occurs")
  
# final code in finally block
finally:
     print("GeeksForGeeks")


# In[30]:


try:  
    a = 10/0  
    print (a)
except ArithmeticError as e:  
        print ("This statement is raising an arithmetic exception.")
        print(e)
else:  
    print ("Success.")


# In[ ]:




