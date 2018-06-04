#coding=utf-8

import numpy as np
import pandas as pd
from pandas import DataFrame

xgb = pd.read_csv("./data/submission.csv")
bn  = pd.read_csv("./data/BernoulliNB.csv")
mn  = pd.read_csv("./data/MultinomialNB.csv")
train_history = pd.read_csv("./Train/Train/history_train.csv")
result_all = pd.merge(pd.merge(xgb,bn,on ='ID'),mn,on ='ID')
result_all['res']= (result_all['xgb']+result_all['BN']+result_all['MN']==3).astype(int)
#print(result_all)
print(train_history.iloc[2,0:1])
#print(train_history.iloc[13,8:13].apply(lambda x: x.sum()))
for i in range(len(result_all)):
    a = train_history.iloc[i,8]+train_history.iloc[i,9]+train_history.iloc[i,10]+train_history.iloc[i,11]+train_history.iloc[i,12]
    b = train_history.iloc[i,13]+train_history.iloc[i,14]+train_history.iloc[i,15]+train_history.iloc[i,16]+train_history.iloc[i,17]
    if result_all.iloc[i,1]==0 and result_all.iloc[i,2]+result_all.iloc[i,3]==2:
        if abs(a)>0:
            if abs(b/a)<0.72:
                result_all.iloc[i,4]=1
    elif result_all.iloc[i,1]==1 and result_all.iloc[i,2]+result_all.iloc[i,3]==1:
        result_all.iloc[i,4]=1
    elif result_all.iloc[i,1]==0 and result_all.iloc[i,2]+result_all.iloc[i,3]==1:
        if result_all.iloc[i,3]==1:
            if train_history.iloc[i,12]>0 and abs(b/a)<0.6:
                result_all.iloc[i,4]=1
        else:
             if train_history.iloc[i,12]>0 and abs(b/a)<0.6:
                result_all.iloc[i,4]=1

np.savetxt('resultall.csv',np.c_[result_all.ID,result_all.res],delimiter=',',header='ID,Default',comments='',fmt='%d')