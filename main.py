# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:38:55 2021

@author: Bram
"""
import pandas as pd
from sklearn.model_selection import KFold
import time
import statistics

from NBTree import NBTree

df = pd.read_csv(r"test2.data")

tree = NBTree(df)
dfcolumns = df.columns
kf = KFold(n_splits = 5)
scores = []
timings = []
predicttimings = []

for train_index, test_index in kf.split(df):
    train, test = df.iloc[train_index], df.iloc[test_index]
    
    start = time.time()
    tree.fit(train)
    end = time.time()
    elapsed = end - start
    timings.append(elapsed)    
    
    start = time.time()
    predictions = tree.predict(test)
    end = time.time()
    elapsed = end - start
    predicttimings.append(elapsed)
    predictions  = predictions.to_list()
    
    score = 0
    for i in range(0, len(predictions)):
        if predictions[i] == test["Decision"].iloc[i]:
            score += 1
    scores.append(score/len(predictions))
    
print("NBTree mean accuracy over 5-fold split: ", statistics.mean(scores))
print("Building NBTree took ", statistics.mean(timings), " on average.")
print("Predicting with NBTree took ", statistics.mean(predicttimings), " on average.")