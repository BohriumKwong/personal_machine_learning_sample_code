#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
import matplotlib.pylab as plt

def performance(labelArr, predictArr):#类标签为int类型
    #labelArr[i] is actual value,predictArr[i] is predict value
    TP = 0.; TN = 0.; FP = 0.; FN = 0.
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1.
        if labelArr[i] == 1 and predictArr[i] == 0:
            FN += 1.
        if labelArr[i] == 0 and predictArr[i] == 1:
            FP += 1.
        if labelArr[i] == 0 and predictArr[i] == 0:
            TN += 1.
    #SN = TP/(TP + FN) #Sensitivity = TP/P  and P = TP + FN
    #SP = TN/(FP + TN) #Specificity = TN/N  and N = TN + FP
    #MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    F1 = 2*TP/(2*TP+FP+FN)
    return F1

if __name__ == '__main__':

    train = pd.read_csv("./Train/Train/train.csv")
    #train.set_index('ID', inplace=True)
    train_xy,val = train_test_split(train, test_size = 0.3,random_state=0)
    # trains_xy,vals = train_test_split(trains, test_size = 0.3,random_state=0)
    Y = train_xy.ACTION
    X = train_xy.drop(['ACTION'],axis=1)
    val_y = val.ACTION
    val_x = val.drop(['ACTION'],axis=1)
    clf = MultinomialNB()
    MultinomialNB(alpha=0.1,  class_prior=False, fit_prior=0.18)
    clf.fit(X, Y)
    # vals['result1'] = clf.predict(val_x)
    # vals.to_csv("./Train/valXX.csv")
    scores = cross_validation.cross_val_score(clf, val_x, val_y, cv=20)
    precisions = cross_validation.cross_val_score(clf, val_x, val_y, cv=20, scoring='precision')
    recalls = cross_validation.cross_val_score(clf, val_x, val_y, cv=20, scoring='recall')
    print(scores)
    #result = clf.predict(val_x)
    #F1 = performance(val_y,np.c_[clf.predict(val_x)])
    F1 = (2*precisions*recalls)/(precisions+recalls)
    #print(F1)
    print('F1分数为:',F1)
    print('F1平均分为:',np.mean(F1),'标准差为:',np.std(F1))
    #########预测
    # # test_basic = pd.read_csv("./PersonalContentDate/basicInfo_test.csv")
    # test_all = pd.read_csv("./Train/Train/newtestXX.csv")
    # bins=[20,25,30,35,40,50,60,70,80]
    # test_all['AGE']=pd.cut(test_all['AGE'],bins,labels=[20,25,30,35,40,50,60,70]).astype("int")
    # bins=[-100,0,0.01,0.2,0.5,0.7,100]
    # # test_all['c05']=(test_all['BILL_AMT6'] - test_all['PAY_AMT5'])/train['CRED_LIMIT']
    # # test_all['c05']=pd.cut(test_all['c05'],bins,labels=[0,0.01,0.2,0.5,0.7,1])
    # # test_all['c05']= test_all['c05'].astype("float")
    # test_all['cc1']=pd.cut(test_all['cc1'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    # test_all['cc2']=pd.cut(test_all['cc2'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    # test_all['cc3']=pd.cut(test_all['cc3'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    # test_all['cc4']=pd.cut(test_all['cc4'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    # test_all['cc5']=pd.cut(test_all['cc5'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    # test_all['PAY6LIT']=pd.cut(test_all['PAY6LIT'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    # test_all['P1']=round(test_all['PAY_AMT6']/1000)
    # test_all['CRED_LIMIT']=round(test_all['CRED_LIMIT']/19500)
    # test_all.PAY_2[test_all.PAY_2==0]=0.5
    # test_all.PAY_2[test_all.PAY_2==-1]=0
    # # test_all.PAY_2[test_all.PAY_2==-2]=0.2
    # test_all.PAY_3[test_all.PAY_3==-1]=0
    # test_all.PAY_3[test_all.PAY_3==0]=0.5
    # # test_all.PAY_3[test_all.PAY_3==-2]=0.2
    # test_all = test_all.drop(['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5',
    #                     'BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
    #                     'PAY_AMT5','PAY_AMT6','PAY_6','PAY_4','PAY_5'],axis=1)
    # test_all[np.isinf(test_all)] = 2
    # test_all[np.isnan(test_all)] = 2
    # test_all[test_all<0]=0.0001
    # preds = clf.predict(test_all.drop(['ID'],axis=1))
    # np.savetxt('./MultinomialNB5.csv',np.c_[test_all.ID,preds],delimiter=',',header='ID,Default',comments='',fmt='%d')
