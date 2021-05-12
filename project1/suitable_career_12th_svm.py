import csv
import math
import random
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from csv import reader
from math import sqrt
from math import exp
from math import pi
from sklearn.externals import joblib
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

lines = csv.reader(open(r'12th - Correct order dataset career prediction  - Sheet1 - Correct order dataset career prediction - Sheet1.csv'))
le = LabelEncoder()  
df = pd.read_csv('12th - Correct order dataset career prediction  - Sheet1 - Correct order dataset career prediction - Sheet1.csv')
TestDatadf = pd.read_csv('12th - Correct order dataset career prediction  - Sheet1 - Correct order dataset career prediction - Sheet1.csv', sep=",", header=None)
df = DataFrame(TestDatadf)
df = pd.DataFrame(TestDatadf)
print(df[0:20])



df[2]= le.fit_transform(df[2])
df[3]= le.fit_transform(df[3])
df[4]= le.fit_transform(df[4])
df[5]= le.fit_transform(df[5])
df[6]= le.fit_transform(df[6])
df[7]= le.fit_transform(df[7])
df[8]= le.fit_transform(df[8])
df[9]= le.fit_transform(df[9]) 
df[10]= le.fit_transform(df[10])
df[11]= le.fit_transform(df[11])
df[12]= le.fit_transform(df[12])
df[13]= le.fit_transform(df[13])
df[14]= le.fit_transform(df[14])
df[34]= le.fit_transform(df[34])
df[35]= le.fit_transform(df[35])
df[36]= le.fit_transform(df[36])
df[37]= le.fit_transform(df[37])
df[38]= le.fit_transform(df[38])
df[39]= le.fit_transform(df[39])
df[40]= le.fit_transform(df[40])
df[41]= le.fit_transform(df[41])
df[42]= le.fit_transform(df[42])
df[43]= le.fit_transform(df[43])
df[44]= le.fit_transform(df[44])
df[45]= le.fit_transform(df[45])
df[46]= le.fit_transform(df[46])
df[47]= le.fit_transform(df[47])
df[48]= le.fit_transform(df[48])
df[49]= le.fit_transform(df[49])
df[50]= le.fit_transform(df[50])
df[51]= le.fit_transform(df[51])
df[54]= le.fit_transform(df[54])

label = df[54]

dataset = df.values.tolist()
for i in range(len(dataset)):
	dataset[i] = [float(x) for x in dataset[i]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.3,random_state=109) 

from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

#filename1 = 'model_new12th_svm.pkl'
#joblib.dump(gnb,filename1)


