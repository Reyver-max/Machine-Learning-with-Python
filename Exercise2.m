!/usr/bin/env python3
 !-*- coding: utf-8 -*-

Created on Sat Feb 20 12:25:15 2021

@author: reyverserna


 
%matplotlib inline 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes_dataset = pd.read_csv("pima-indians-diabetes.csv")

diabetes_dataset.shape
diabetes_dataset.describe()
diabetes_dataset.groupby("Outcome").size()


dataset_nozeros = diabetes_dataset.copy()

zero_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] 
diabetes_dataset(zero_fields) = diabetes_dataset[zero_fields].replace(0, np.nan)
diabetes_dataset(zero_fields) = diabetes_dataset[zero_fields].fillna(dataset_nozeros.mean())
diabetes_dataset.describe()  
from sklearn.model_selection import train_test_split 

 divide into training and testing data
train,test = train_test_split(diabetes_dataset, test_size == 0.25, random_state==0, stratify == diabetes_dataset['Outcome']) 

 separate the 'Outcome' column from training/testing data
train_X = train(train.columns(:8))
test_X = test(test.columns(:8))
train_Y = train('Outcome')
test_Y = test('Outcome')

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train_X,train_Y)
prediction = model.predict(test_X)

!calculate accuracy
from sklearn import metrics

print(metrics.accuracy_score(test_Y, prediction))

the_most_outcome = diabetes_dataset('Outcome').median()
prediction2 = [the_most_outcome for i in range(len(test_Y))]
print(metrics.accuracy_score(test_Y, prediction2))

!histogram of predicted probabilities

save_predictions_proba = model.predict_proba(test_X)[:, 1]  # column 1

plt.hist(save_predictions_proba, bins=10)
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')
plt.show()

