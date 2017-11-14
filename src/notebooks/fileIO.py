
# coding: utf-8

# In[4]:


import os


# In[5]:


# create a dictionary that maps a file name to a species
def readFileToDic():
    curDic = {}
    with open('../../fileNameToSpecies.txt','r') as file1:
        line1 = file1.readline().strip()
        while line1:
            line1_sp = line1.split('\t')
            fileName = line1_sp[0]
            species = line1_sp[1]
            curDic[fileName] = species
            line1=file1.readline().strip()
    return(curDic)

