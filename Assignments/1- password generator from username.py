#!/usr/bin/env python
# coding: utf-8
This is a simple app using python to generate password from first names.


# In[2]:



import random
import string


# In[34]:


print("hello! to password generator from username")

def generate_password(name):
    #convert username to lowercase
    lower = name.lower()
    
    #convert username to uppercase
    upper = name.upper()
    
    #store number and symbols
    num = string.digits
    symbols = string.punctuation
    
    all =  num + symbols

    # Generate random 4 lower character from username 
    password1 = ''.join(random.choice(lower) for i in range(4))
    # Generate random 4 upper character from username 
    password2 = ''.join(random.choice(upper) for i in range(4))
    # Generate random 3 number and symbols 
    password3 = ''.join(random.choice(all) for i in range(3))

    #Concatenate the edited username with the random numbers and symbols and return the generated password
    password = password1 + password2 + password3
    return password




# In[37]:


#input your name
name = str(input("Please Enter your username : "))
print("your password is " + generate_password(name))


# In[ ]:




