# -*- coding: utf-8 -*-
"""Size_Shirt_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ej0yKSq5oGxlbF2qyvAsZmzWILbhq0xT
"""

import numpy as np
import pandas as pd

df = pd.read_csv('sizes_data.csv')

df.describe()

df = df.dropna()
df

df.dtypes

from sklearn.model_selection import train_test_split

X = df.iloc[:,:-1]
y = df['size']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=109)

y = df.select_dtypes(include=[object])
y.head()

"""# Ridge Classifier"""

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import RidgeClassifier

clf = RidgeClassifier()

# Reshape y using ravel()
y = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=109)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

baseline_performance = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy with without feature selection: {baseline_performance}')

sample = pd.DataFrame([[40, 23, 180]], columns=['weight', 'age', 'height'])
output= clf.predict(sample)
print(output)

import joblib
joblib.dump(clf,'model.pkl')

import streamlit as st
import joblib
import pandas as pd