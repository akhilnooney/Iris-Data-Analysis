#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris() #To load Iris Dataset
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['species'])#convert data to Dataframe
df['species']=df['species'].replace(0,'setosa')
df['species']=df['species'].replace(1,'versicolor')
df['species']=df['species'].replace(2,'virginica')
df['species']=(df['species']=='setosa').astype(int)*1+(df['species']=='versicolor').astype(int)*2+(df['species']=='virginica').astype(int)*3
x=df.drop(['species'],axis=1)
y=df['species']
x_training_data,x_testing_data,y_training_data,y_testing_data=train_test_split(x,y,test_size=0.33,random_state=42)#Train and Test data
x_training_data,x_testing_data,y_training_data,y_testing_data=np.array(x_training_data),np.array(x_testing_data),np.array(y_training_data),np.array(y_testing_data)
def K_score(n):#Function definition
    Model=KNeighborsClassifier(n_neighbors=n).fit(x_training_data,y_training_data)#model fitting
    global prediction
    prediction=Model.predict(x_testing_data)#Model prediction
K_score(4)#Function call
print("Accuracy Score before validation: {}".format(accuracy_score(y_testing_data, prediction)))
neighbors = list(range(1,50,2))
cross_validation_scores = []
Missclassification_error=[]
for K in neighbors:# perform 10-fold cross validation
    knn = KNeighborsClassifier(n_neighbors=K)#Model 
    score = cross_val_score(knn, x_training_data, y_training_data, cv=10, scoring='accuracy')#perform Crossvalidation 
    cross_validation_scores.append(score.mean())#Mean calculation for all folds
#Missclassification_error = [1 - y for y in cross_validation_scores]
for y in cross_validation_scores:
    Missclassification_error.append(1-y)
K_Value = neighbors[Missclassification_error .index(min(Missclassification_error ))]#Best K-value
print("Best K value :{}".format(K_Value))
K_score(K_Value)#Function Call
print("Accuracy Model after Cross-Validation: {}".format(accuracy_score(y_testing_data, prediction)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




