# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:30:38 2021

@author: Sander van Houtert
@author: Bram Vogels
"""

import pandas as pd
import numpy as np
import sklearn.naive_bayes as skn
from sklearn.model_selection import KFold
from sklearn import preprocessing
import statistics
import time

class Node:
    def __init__(self, numAttr, attributes, splitAttr, splitVal, parent):
        self.numAttr = numAttr
        self.attributes = attributes
        self.splitAttr = splitAttr
        self.splitVal = splitVal
        self.children = []
        self.parent = parent
        self.nbc = skn.GaussianNB()
        self.isLeaf = False
        
    def trainClassifier(self, d):
        data = createSubset(d, self.attributes, self)
        nData = data.to_numpy()
        enc = preprocessing.OrdinalEncoder()
        enc.fit(nData[:, 0:nData.shape[1]-1])
        nData[:, 0:nData.shape[1]-1] = enc.transform(nData[:, 0:nData.shape[1]-1])
        X = nData[:, 0:nData.shape[1]-1]
        y = nData[:, nData.shape[1]-1].astype('int')
        self.nbc.fit(X, y)
        
    def printNode(self):
        print("isLeaf: ", self.isLeaf)
        print("splitVal: ", self.splitVal)
        print("splitAttr: ", self.splitAttr)

    def predict(self, d):
        #print("predicting class of: ", d)
        return self.nbc.predict(d)
        
class NBTree:
    def __init__(self, df):
        self.data = df
        self.numAttr = -1
        self.attributes = []
        self.rootNode = None

    def fit(self, df):
        #Generate tree based on given dataframe.
        self.numAttr = len(df.columns)-1
        self.attributes = df.columns[0:len(df.columns)-1]
        #Create the root node.
        self.rootNode = Node(self.numAttr, self.attributes, None, None, self)
        #Find the rootnode's children through a recursive step.
        self.rootNode.children = self.recursiveBuildTree(df, self.numAttr, self.attributes, self.rootNode, None)
    
    def recursiveBuildTree(self, d, numAttributes, attributes, parent, split):
        if(isinstance(parent, Node)):
            parent.splitAttr = split
        #C4.5 step to weed out edge-cases
        if(self.allSameClass(d)):
            parent.isLeaf = True
            parent.trainClassifier(d)
            return []
        best, errSplit = findAttr(d, attributes, parent)
        errLeaf = findErrLeaf(d, attributes, parent)
        #if split is at least 5% better than a regular Naïve-Bayes classifier, create new nodes.
        if(errSplit/errLeaf <= 0.95):
            children = []
            newAttributes = attributes.drop(best)
            for i in d[best].unique():
                node = Node(numAttributes-1, newAttributes, None, i, parent)
                #Recursive step, define children of current node.
                node.children = self.recursiveBuildTree(d, numAttributes-1, newAttributes, node, best)
                children.append(node)
            for j in children:
                j.printNode()
            return children
        #if split doesn't result in enough improvement define current node as leaf and train it's classifier on the resulting dataset.
        else:
            parent.isLeaf = True
            parent.trainClassifier(d)
            return []
        
    def allSameClass(self, d):
        if d["Decision"].nunique() == 1:
            return True
        else:
            return False
        
    def getRoot(self):
        return self.rootNode
    
    def predictScore(self, d):
        leaf = self.findLeaf(d)
        dPrimed = primeData(d, leaf.attributes)
        nData = createSubset(self.data, leaf.attributes, leaf).to_numpy()
        nd = dPrimed.to_numpy()
        nd = nd.reshape(1, -1)
        enc = preprocessing.OrdinalEncoder()
        enc.fit(nData[:, 0:nData.shape[1]-1])
        nd = enc.transform(nd)
        return leaf.predict(nd)    
    
    def predict(self, d):
        return d.apply(self.predictScore, axis = 1)


    def findLeaf(self, d):
        node = self.getRoot()
        tempNode = node
        while(tempNode.isLeaf is False):
            for i in tempNode.children:
                if(d[i.splitAttr] == i.splitVal):
                    tempNode = i
        return tempNode
        
def createSubset(d, attributes, node):
    if(isinstance(node.parent, NBTree) is True):
        return d
    tempNode = node
    subset = d
    while(tempNode.splitAttr != None):
        subset = subset[subset[tempNode.splitAttr] == tempNode.splitVal].drop(tempNode.splitAttr, axis=1)
        tempNode = tempNode.parent
    return subset

def primeData(d, attributes):
    for i in d.index:
        if i not in attributes:
            d = d.drop(i, axis=0)
    return d

def findErrLeaf(d, attributes, node):
    data = createSubset(d, attributes, node)
    nData = data.to_numpy()
    enc = preprocessing.OrdinalEncoder()
    enc.fit(nData[:, 0:nData.shape[1]-1])
    nData[:, 0:nData.shape[1]-1] = enc.transform(nData[:, 0:nData.shape[1]-1])
    kf = KFold(n_splits = 5)
    X = nData[:, 0:nData.shape[1]-2]
    y = nData[:, nData.shape[1]-1].astype('int')
    scores = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = skn.GaussianNB()
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    return 1-statistics.mean(scores)

def findAttr(d, attributes, node):
    #Create a subset based on the remaining attributes and previous splits
    data = createSubset(d, attributes, node)
    #Keep track of the raw accuracy scores and weighted utility scores per attribute
    accuracies = []
    attrScores = []
    #Calculate the accuracies of applying Naive-Bayes classifier for each leaf resulting in a split on said attribute
    for i in attributes:
        relevant = True
        weightedValueScores = []
        unweightedValueScores = []
        #Calculate accuracy for each possible leaf resulting from split
        for j in data[i].unique():
            #If split results in node with less than 30 elements, consider the attribute irrelevant for potential split
            if(data[data[i] == j].shape[0] >= 30):
                tempData = data[data[i] == j]
                scores = []
                nData = tempData.to_numpy()
                enc = preprocessing.OrdinalEncoder()
                enc.fit(nData[:, 0:nData.shape[1]-1])
                nData[:, 0:nData.shape[1]-1] = enc.transform(nData[:, 0:nData.shape[1]-1])
                kf = KFold(n_splits = 5)
                X = nData[:, 0:nData.shape[1]-1]
                y = nData[:, nData.shape[1]-1].astype('int')
                for train_index, test_index in kf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    clf = skn.GaussianNB()
                    clf.fit(X_train, y_train)
                    scores.append(clf.score(X_test, y_test))
                weightedValueScores.append(statistics.mean(scores) * tempData.shape[0] / d.shape[0])
                unweightedValueScores.append(statistics.mean(scores))
            else:
                relevant = False
        if relevant is True:
            attrScores.append(statistics.mean(weightedValueScores))
            accuracies.append(statistics.mean(unweightedValueScores))
        else:
            attrScores.append(0)
            accuracies.append(0)
    #Return label of best attribute to split on and corresponding unweighted mean accuracy of resulting leaves.
    return data.columns[np.argmax(attrScores)], (1-accuracies[np.argmax(attrScores)])