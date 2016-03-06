# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:22:31 2016

@author: yatwong
"""

#import sys
#import os
import time
from sklearn import metrics
from sklearn import datasets
#import numpy as np
#import cPickle as pickle



# 朴素贝叶斯
def naive_bayes_classifier(trainX, trainY):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(trainX, trainY)
    return model

# KNN
def knn_classifier(trainX, trainY):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(trainX, trainY)
    return model

# 逻辑回归
def logistic_regression_classifier(trainX, trainY):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(trainX, trainY)
    return model

# 随机森林
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model

# 决策树
def decision_tree_classifier(trainX, trainY):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(trainX, trainY)
    return model

#svm
def svm_classifier(trainX, trainY):
    from sklearn.svm import SVC
    model = SVC(gamma=0.001, C=100.)
    model.fit(trainX, trainY)
    return model

class Traverse2Dlist():
    def __init__(self, twoDList):
        self.listLen = len(twoDList)
        self.idx = [0 for i in range(self.listLen)]
        self.twoDList = twoDList
        return
    def getNext(self):
        pointer = self.listLen - 1
        while True:
            if self.idx[pointer]+1 > len(self.twoDList[pointer])-1:
                self.idx[pointer] = 0
                pointer -= 1
                if pointer == -1:
                    return None
            else:
                self.idx[pointer] += 1
                return self.idx

def svm_classifier_auto(trainX, trainY, testX, testY):
    from sklearn.svm import SVC
    parametersList = [
        [x*0.0001 for x in range(1,101,10)],#gamma
        [x for x in range(10, 200, 10)]#C
    ]
    travese2DList = Traverse2Dlist(parametersList)
    idx = travese2DList.getNext()
    bestAccuracy = 0.
    bestIdx=[]
    worstAccuracy = 1.
    while idx != None:
        model = SVC(gamma=parametersList[0][idx[0]], \
        C=parametersList[1][idx[1]])
        model.fit(trainX, trainY)
        predict = model.predict(testX)
        accuracy = metrics.accuracy_score(testY, predict)
        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestIdx = idx
        if accuracy < worstAccuracy:
            worstAccuracy = accuracy
        idx = travese2DList.getNext()
        
    return bestAccuracy, 'gamma='+ str(parametersList[0][bestIdx[0]])\
    +' C='+ str(parametersList[1][bestIdx[1]]), worstAccuracy


if __name__ == '__main__':
    
    test_classifiers = ['朴素贝叶斯', 'KNN', 'LR', '随机森林', '决策树', 'SVM']
    classifiers = {
        '朴素贝叶斯':naive_bayes_classifier,
        'KNN':knn_classifier,
        'LR':logistic_regression_classifier,
        '随机森林':random_forest_classifier,
        '决策树':decision_tree_classifier,
        'SVM':svm_classifier,
        #'SVMCV':svm_cross_validation,
        #'GBDT':gradient_boosting_classifier
    }
    print("读取数据...")
    digits = datasets.load_digits();
    trainX = digits.data[:1200]
    trainY = digits.target[:1200]
    testX = digits.data[1200:]
    testY = digits.target[1200:]
    
    for key in classifiers:
        print("---使用%s---" % key)
        model = classifiers[key](trainX, trainY)
        predict = model.predict(testX)
        accuracy = metrics.accuracy_score(testY, predict)
        print('\t准确率: %.2f%%' % (100 * accuracy))

    print("---穷举svm部分参数,获得最佳参数---")
    startTime = time.time()
    best, s, worst = svm_classifier_auto(trainX, trainY, testX, testY)
    print('\t最佳准确率: %.2f%%' % (100 * best))
    print('\t最佳参数:', s)
    print('\t最差准确率: %.2f%%' % (100 * worst))
    print('\t用时: %ds' % (time.time()-startTime))