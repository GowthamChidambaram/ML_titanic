import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#printing some basic informations
train=pd.read_csv("titanic_train.csv")
print(train.head())
print(train.columns)
print(train.describe())
print(train.info())

#creating heat map to find null valued columns
print(train.isnull())
sns.heatmap(data=train.isnull(),yticklabels=False,cmap="viridis",cbar=False)
plt.show()

#some data visualisation
sns.set_style("whitegrid")
sns.countplot(x="Survived",data=train)
plt.show()
sns.countplot(x="Survived",data=train,hue="Pclass")
plt.show()
sns.countplot(x="Survived",data=train,hue="Sex")
plt.show()

sns.distplot(train["Age"].dropna(),bins=35,kde=False)
plt.tight_layout()
plt.show()
sns.countplot(x="SibSp",data=train)
plt.show()
sns.countplot(x="Parch",data=train)
plt.show()
train["Fare"].hist(bins=35)
plt.show()

#filling up nan values with the mean age of resp class
clas1_mean=train[train["Pclass"]==1]["Age"].mean()
print(clas1_mean)
clas2_mean=train[train["Pclass"]==2]["Age"].mean()
print(clas2_mean)
clas3_mean=train[train["Pclass"]==3]["Age"].mean()
print(clas3_mean)
def impute(cols):
    age=cols[0]
    clas=cols[1]
    if pd.isnull(age):
        if clas==1:
            return clas1_mean
        elif clas==2:
            return clas2_mean
        else :
            return clas3_mean
    else:
        return age

train["Age"]=train[["Age","Pclass"]].apply(impute,axis=1)

sns.heatmap(data=train.isnull(),yticklabels=False,cmap="viridis",cbar=False)
plt.show()

#dropping cabin column
train.drop("Cabin",axis=1,inplace=True)
print(train.head())
train.dropna(inplace=True)
sns.heatmap(data=train.isnull(),yticklabels=False,cmap="viridis",cbar=False)
plt.show()

#coverting other string value columns to numbered values(charecterising)
sex=pd.get_dummies(train["Sex"],drop_first=True)
embark=pd.get_dummies(train["Embarked"],drop_first=True)
train=pd.concat([train,sex,embark],axis=1)
print(train.head())
print(train.info())

#splitting the train dataset into test and train datasets
x=train[["Pclass","Age","SibSp","Parch","Fare","male","Q","S"]]
y=train["Survived"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
predictions=logmodel.predict(x_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


