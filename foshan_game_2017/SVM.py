#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import matplotlib.pylab as plt


# def performance(labelArr, predictArr):#类标签为int类型
#     #labelArr[i] is actual value,predictArr[i] is predict value
#     TP = 0.; TN = 0.; FP = 0.; FN = 0.
#     for i in range(len(labelArr)):
#         if labelArr[i] == 1 and predictArr[i] == 1:
#             TP += 1.
#         if labelArr[i] == 1 and predictArr[i] == 0:
#             FN += 1.
#         if labelArr[i] == 0 and predictArr[i] == 1:
#             FP += 1.
#         if labelArr[i] == 0 and predictArr[i] == 0:
#             TN += 1.
#     #SN = TP/(TP + FN) #Sensitivity = TP/P  and P = TP + FN
#     #SP = TN/(FP + TN) #Specificity = TN/N  and N = TN + FP
#     #MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
#     F1 =  2*TP/(2*TP+FP+FN)
#     return F1

if __name__ == '__main__':

    train_basic = pd.read_csv("./Train/Train/basicInfo_train.csv")
    train_history = pd.read_csv("./Train/Train/history_train.csv")
    train_default = pd.read_csv("./Train/Train/default_train.csv")
    train = pd.merge(pd.merge(train_basic,train_history,on ='ID'),train_default,on ='ID')
    train.set_index('ID', inplace = True)

    train = train.drop(['BILL_AMT1','PAY_3','PAY_4','PAY_5'],axis=1)
    pd.qcut(train.AGE,5)
    train_xy,val = train_test_split(train, test_size = 0.4,random_state=0)
    Y = train_xy.Default
    X = train_xy.drop(['Default'],axis=1)
    val_y = val.Default
    val_x = val.drop(['Default'],axis=1)
    #xgb_val = xgb.DMatrix(val_x,label=val_y)
    #xgb_train = xgb.DMatrix(X, label=Y)
    #watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]
    #clf = svm.SVC(kernel='rbf', C=1)
    clf = RandomForestClassifier(n_estimators=150)
    clf.fit(X, Y)
    scores = cross_validation.cross_val_score(clf, val_x, val_y, cv=2)
    precisions = cross_validation.cross_val_score(clf, val_x, val_y, cv=2, scoring='precision')
    recalls = cross_validation.cross_val_score(clf, val_x, val_y, cv=2, scoring='recall')
    print(scores)
    #result = clf.predict(val_x)
    #F1 = performance(val_y,np.c_[clf.predict(val_x)])
    F1 = (2*precisions*recalls)/(precisions+recalls)
    #print(F1)
    print('F1分数为:',F1)