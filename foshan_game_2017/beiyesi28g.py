#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
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

    train = pd.read_csv("./Train/newtrain2.csv")
    train.set_index('ID', inplace=True)
    train=train[train['BILL_AMT1']+train['BILL_AMT2']+train['BILL_AMT3']+train['BILL_AMT4']+train['BILL_AMT5']
    +train['BILL_AMT6']+train['PAY_AMT1']+train['PAY_AMT2']+train['PAY_AMT3']+train['PAY_AMT4']+train['PAY_AMT5']
    +train['PAY_AMT6'] -train['Default'] !=-1]
    # train['sub']= (train['BILL_AMT2']+train['BILL_AMT3']+train['BILL_AMT4']+train['BILL_AMT5']
    # +train['BILL_AMT6']-(train['PAY_AMT1']+train['PAY_AMT2']+train['PAY_AMT3']+train['PAY_AMT4']
    #                      +train['PAY_AMT5'])>=0).astype(float)
    bins=[20,25,30,35,40,45,50,55,60,65,70,75,80]
    train['AGE']=pd.cut(train['AGE'],bins,labels=[0.21,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75]).astype("float")
    bins=[-100,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,100]
    # train['c00']=(train['BILL_AMT1'] - train['PAY_AMT6'])/train['CRED_LIMIT']
    # train['c00']=pd.cut(train['c00'],bins,labels=[-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    # train['c00']= train['c00'].astype("int")
    train['c01']=(train['BILL_AMT2'] - train['PAY_AMT1'])/train['CRED_LIMIT']
    train['c01']=pd.cut(train['c01'],bins,labels=[-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    train['c01']= train['c01'].astype("int")
    train['c02']=(train['BILL_AMT3'] - train['PAY_AMT2'])/train['CRED_LIMIT']
    train['c02']=pd.cut(train['c02'],bins,labels=[-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    train['c02']= train['c02'].astype("int")
    train['c03']=(train['BILL_AMT4'] - train['PAY_AMT3'])/train['CRED_LIMIT']
    train['c03']=pd.cut(train['c03'],bins,labels=[-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    train['c03']= train['c03'].astype("int")
    train['c04']=(train['BILL_AMT5'] - train['PAY_AMT4'])/train['CRED_LIMIT']
    train['c04']=pd.cut(train['c04'],bins,labels=[-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    train['c04']= train['c04'].astype("int")
    train['c05']=(train['BILL_AMT6'] - train['PAY_AMT5'])/train['CRED_LIMIT']
    train['c05']=pd.cut(train['c05'],bins,labels=[-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    train['c05']= train['c05'].astype("int")
    #########################
    bins=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,100]
    train['cc1']=pd.cut(train['cc1'],bins,labels=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).astype("int")
    train['cc2']=pd.cut(train['cc2'],bins,labels=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).astype("int")
    train['cc3']=pd.cut(train['cc3'],bins,labels=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).astype("int")
    train['cc4']=pd.cut(train['cc4'],bins,labels=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).astype("int")
    train['cc5']=pd.cut(train['cc5'],bins,labels=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).astype("int")
    #train['PAY6LIT']=pd.cut(train['PAY6LIT'],bins,labels=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).astype("int")
    train['B1']=round(train['BILL_AMT1']/79000,1)
    #train['CRED_LIMIT']=round(train['CRED_LIMIT']/144000)
    train['sumC']=round((train['sumC'])/10,1)
    train = train.drop(['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5',
                        'BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
                        'PAY_AMT5','PAY_AMT6','PAY_3','PAY_4','PAY_5','PAY6LIT','CRED_LIMIT'],axis=1)
    #train[train<0]=0
    train_xy,val = train_test_split(train, test_size = 0.3,random_state=0)
    Y = train_xy.Default
    X = train_xy.drop(['Default'],axis=1)
    val_y = val.Default
    val_x = val.drop(['Default'],axis=1)
    clf = BernoulliNB()
    BernoulliNB(alpha=0.1, binarize=0.35, class_prior=False, fit_prior=0.18)
    clf.fit(X, Y)
    scores = cross_validation.cross_val_score(clf, val_x, val_y, cv=15)
    precisions = cross_validation.cross_val_score(clf, val_x, val_y, cv=15, scoring='precision')
    recalls = cross_validation.cross_val_score(clf, val_x, val_y, cv=15, scoring='recall')
    print(scores)
    #result = clf.predict(val_x)
    #F1 = performance(val_y,np.c_[clf.predict(val_x)])
    F1 = (2*precisions*recalls)/(precisions+recalls)
    #print(F1)
    print('F1分数为:',F1)
    print('F1平均分为:',np.mean(F1),'标准差为:',np.std(F1))
    #########预测
    # test_basic = pd.read_csv("./PersonalContentDate/basicInfo_test.csv")
    # test_history = pd.read_csv("./PersonalContentDate/history_test.csv")
    # test_all = pd.merge(test_basic,test_history,on ='ID')
    # bins=[20,25,30,35,40,45,50,55,60,65,70,75,80]
    # test_all['AGE']=pd.cut(test_all['AGE'],bins,labels=[0.21,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75]).astype("float")
    # bins=[-100,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,100]
    # test_all['c00']=(test_all['BILL_AMT1'] - test_all['PAY_AMT6'])/test_all['CRED_LIMIT']
    # test_all['c00']=pd.cut(test_all['c00'],bins,labels=[-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    # test_all['c00']= test_all['c00'].astype("int")
    # test_all['c01']=(test_all['BILL_AMT2'] - test_all['PAY_AMT1'])/test_all['CRED_LIMIT']
    # test_all['c01']=pd.cut(test_all['c01'],bins,labels=[-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    # test_all['c01']= test_all['c01'].astype("int")
    # test_all['c02']=(test_all['BILL_AMT3'] - test_all['PAY_AMT2'])/test_all['CRED_LIMIT']
    # test_all['c02']=pd.cut(test_all['c02'],bins,labels=[-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    # test_all['c02']= test_all['c02'].astype("int")
    # test_all['c03']=(test_all['BILL_AMT4'] - test_all['PAY_AMT3'])/test_all['CRED_LIMIT']
    # test_all['c03']=pd.cut(test_all['c03'],bins,labels=[-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    # test_all['c03']= test_all['c03'].astype("int")
    # test_all['c04']=(test_all['BILL_AMT5'] - test_all['PAY_AMT4'])/test_all['CRED_LIMIT']
    # test_all['c04']=pd.cut(test_all['c04'],bins,labels=[-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    # test_all['c04']= test_all['c04'].astype("int")
    # test_all['c05']=(test_all['BILL_AMT6'] - test_all['PAY_AMT5'])/test_all['CRED_LIMIT']
    # test_all['c05']=pd.cut(test_all['c05'],bins,labels=[-1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    # test_all['c05']= test_all['c05'].astype("int")
    # test_all['B1']=round(test_all['BILL_AMT1']/104155)
    # test_all['CRED_LIMIT']=round(test_all['CRED_LIMIT']/144000)
    # test_all = test_all.drop(['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5',
    #                     'BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
    #                     'PAY_AMT5','PAY_AMT6'],axis=1)
    # preds = clf.predict(test_all.drop(['ID'],axis=1))
    # np.savetxt('BernoulliNB.csv',np.c_[test_all.ID,preds],delimiter=',',header='ID,Default',comments='',fmt='%d')