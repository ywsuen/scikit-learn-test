# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:22:31 2016

@author: yatwong
"""

import sys
import os
import time
from sklearn import metrics
from sklearn import datasets
import numpy as np
#import cPickle as pickle

import adultDataset2array

# 朴素贝叶斯
def naive_bayes_classifier(trainX, trainY):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(trainX, trainY)
    return model

#svm
def svm_classifier(trainX, trainY):
    from sklearn.svm import SVC
    model = SVC(gamma=0.001, C=100.)
    model.fit(trainX, trainY)
    return model



if __name__ == '__main__':
    data_file=""
    tresh = 0.5
    
    test_classifiers = ['朴素贝叶斯','SVM']
    classifiers = {'朴素贝叶斯':naive_bayes_classifier,
                  #'KNN':knn_classifier,
                   #'LR':logistic_regression_classifier,
                  #'RF':random_forest_classifier,
                   #'DT':decision_tree_classifier,
                  'SVM':svm_classifier,
                #'SVMCV':svm_cross_validation,
                 #'GBDT':gradient_boosting_classifier
    }
    print("读取数据...")
    #trainX, trainY, testX, testY = adultDataset2array.getData()
    digits = datasets.load_digits();
    trainX = digits.data[:1200]
    trainY = digits.target[:1200]
    testX = digits.data[1200:]
    testY = digits.target[1200:]
    #print("****************************数据预览****************************")
    #print("\n训练的X\n", trainX, "\n训练的Y\n", trainY, "\n测试的X\n",\
    #testX, "\n测试的Y\n", testY)
    #print("**************************数据预览完毕***************************")
    
    for key in classifiers:#获得字典key
        print("---使用%s---" % key)
        model = classifiers[key](trainX, trainY)
        predict = model.predict(testX)
        accuracy = metrics.accuracy_score(testY, predict)
        print('准确率: %.2f%%' % (100 * accuracy))

    
    