# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:02:17 2019

@author: kdandebo
"""

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 500)
df = pd.ExcelFile('C:/Users/kdandebo/Desktop/HomelatoptoKarthiklaptop/Python/datasetforpractice/sms_raw_NB.xlsx')
df = df.parse("sms_raw_NB")
print(df.head(10))
print(df.columns)
type(df)
print(df.head(10))
#str(df['type'])
df['type'].describe()
df['text'].describe()
x = df['text'];
y = df['type'];

#splitting to train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
x_train.shape
x_test.shape
y_train.shape

print(x_train.head(10))

#convert to lowercase becuase the Tfid fit and transofmr only accepts in lowercase
x_train = x_train.apply(lambda x: " ".join(x.lower() for x in str(x).split()))
x_train.head(10)

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
x_train_trfm = vect.fit_transform(x_train)
x_train_trfm

#only applt transoform not fit for test
x_test_trfm = vect.transform(x_test)
x_test_trfm


#getting all the individual list of words from the vectorization process
 
features_name = vect.get_feature_names()


features_name


#using only the most freq words by using the percentile function, asking it to filter all words which are repeated atleast 5 times

from sklearn.feature_selection import SelectPercentile

Selector = SelectPercentile(percentile = 5)

#fitting that percentile function to training data
Selector.fit(x_train_trfm,y_train)

#we are converting it to array , becuase the input needs to be in aray format to input into Naive bayes algorithm
x_train_trfm = Selector.transform(x_train_trfm).toarray()
x_test_trfm = Selector.transform(x_test_trfm).toarray()
x_train_trfm

#Now implement Naive bayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
#gaussian naive bayes, works great for cont category, but although we are using here
G = GaussianNB();
G.fit(x_train_trfm,y_train)
y_pred = G.predict(x_test_trfm)
y_pred
y_test

#accuracy
from sklearn import metrics
accu = metrics.accuracy_score(y_test,y_pred)
print(accu)


#BernoullisNB
#it is used when multile variate model , similar to multinomial naive baye, works great for dscrete category
B = BernoulliNB();
B.fit(x_train_trfm,y_train)
y_pred1 = B.predict(x_test_trfm)
y_pred1

#accuracy
from sklearn import metrics
accu1 = metrics.accuracy_score(y_test,y_pred1)
print(accu1)


M = MultinomialNB();
M.fit(x_train_trfm,y_train)
y_pred2 = M.predict(x_test_trfm)
y_pred2


mess1 = pd.Series('Ultimate Spider-man game (å£4.50) on ur mobile right now   Text SPIDER to 83338 for the game & we ll send u a FREE 8Ball wallpaper')
mess1_trfm = vect.transform(mess1)
mess1_trfm = Selector.transform(mess1_trfm).toarray()
B.predict(mess1_trfm)

