# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:18:53 2021

@author: Admin
"""

#Name: Thomas Amal Ambrose
import numpy as np
import pandas as pd
#importing the dataset
D=pd.read_csv("C:/Users/Admin/Desktop/Project/customer-segmentation-dataset/Mall_Customers.csv")
D.columns
D.columns=['CustomerID', 'Gender', 'Age', 'Annual_Income(k$)','Spending_Score(1-100)']
D.isna().sum()
#Column gender should be label encoded
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
D['Gender']=le.fit_transform(D['Gender'])
#customer Id is not needed so dropping that one
D_new=D.drop(['CustomerID'], axis=1)

#Visualizing Gender and age distribution
import matplotlib.pyplot as plt
plt.scatter(D_new.Age, D_new.Gender)
plt.scatter(D_new.Age,D_new['Annual_Income(k$)'])
plt.scatter(D_new.Age,D_new['Spending_Score(1-100)'])

#barplot
plt.bar(D_new.Age, height=np.arange(1,201,1))
#Barplot of Age shows that data covers all major customers from age 20-70
plt.hist(D_new.Gender)
#Histogram shows we have more female customer data
plt.hist(D_new['Annual_Income(k$)'])
plt.hist(D_new['Spending_Score(1-100)'])
import seaborn as sns
#data distribution using boxplot
sns.boxplot(D_new.Age)#0 outliers
sns.boxplot(D_new.Gender)#0 outliers
sns.boxplot(D_new['Annual_Income(k$)'])#one outlier present
sns.boxplot(D_new['Spending_Score(1-100)'])#no outlier

#Outlier treatment
IQR=D_new['Annual_Income(k$)'].quantile(0.75)-D_new['Annual_Income(k$)'].quantile(0.25)
lower_limit=D_new['Annual_Income(k$)'].quantile(0.25)-1.5*(IQR)
lower_limit#-13.25, Income wont become -ve, so passiing value 0 as lower limit
lower_limit=0
upper_limit=D_new['Annual_Income(k$)'].quantile(0.75)+1.5*(IQR)

D_new['Annual_Income(k$)']=(np.where(D_new['Annual_Income(k$)']>upper_limit,upper_limit, np.where(D_new['Annual_Income(k$)']<lower_limit,lower_limit,D_new['Annual_Income(k$)'])))
sns.boxplot(D_new['Annual_Income(k$)']),plt.title('Boxplot'),plt.show()#no outliers

D_new.describe()
#normalising the data
def norm(x):
    z=(x-x.min())/(x.max()-x.min())
    return z

D_norm=norm(D_new)
D_norm.describe()
#DAta is ready for operations
#KMeans clustering
from sklearn.cluster import KMeans
plt.scatter(D_norm['Annual_Income(k$)'], D_norm['Spending_Score(1-100)'])
#Data is spread all over the areas and has a high density at area of 40<->60
TWS=[]#This is used to store total within sum of squares while implementing kmeans with random no of n_clusters
k=list(range(2,9))
for i in k:
    km=KMeans(n_clusters=i)
    km.fit(D_norm)
    TWS.append(km.inertia_)

TWS#Total Wthin sum of square values is used to do the elbow curve
#To minimize the vedio I didnt explained this one on my demo vedio
#Elbow curve(to find optimal n_cluster)
plt.plot(k,TWS,'ro-');plt.xlabel('No_of_Cluster');plt.ylabel('SSE')
#From the plot number clusters should be=4

km=KMeans(n_clusters=4)
km.fit(D_norm)
km.labels_
c_lab=pd.Series(km.labels_)#Cluster labels 
D_new['Cluster']=c_lab

C_Dat=D_new.iloc[:,[4,0,1,2,3]]
C_Dat.groupby(C_Dat.Cluster).mean()
#We got 4 clusters
"The cluster 3 of average age=28 ,avg_annual income=60k$, has an average_spending_score of 67.68"
"Cluster 2 has avg_age 28, avg_annual income=61k$ , has avg_spending_score of 71.67"

"More the spending scores more they spent on products"

"I noticed impact of age eventhough the avg salary is 62k$ and 58k$ of cluster 0 and 1 their spending scores is so low to 29.208 and 34.7818"
"Male spends more at the mall in avg age of 28 and in avg age 49 Female spends more"

C_Dat.to_csv("K-Means_Mall_Customers.csv", index=False)


