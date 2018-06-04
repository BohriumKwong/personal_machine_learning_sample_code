#coding=utf-8

import numpy as np
import pandas as pd
from pandas import DataFrame

xgb = pd.read_csv("./data/xgbxx.csv")
mn  = pd.read_csv("./data/MultinomialNB3.csv")
train_history = pd.read_csv("./PersonalContentDate/history_test.csv")
result_all = pd.merge(pd.merge(mn,xgb,on ='ID'),train_history,on ='ID')
# result_all['res']=result_all['Default']
# result_all['res']= (result_all['xgb']+result_all['BN']+result_all['MN']==3).astype(int)
#print(result_all)
#print(train_history.iloc[2,0:1])
#print(train_history.iloc[13,8:13].apply(lambda x: x.sum()))
for i in range(len(train_history)):
    if result_all.iloc[i,1]==1 and sum(result_all.iloc[i,9:15])<=0:
        result_all.iloc[i,1]=0
    if result_all.iloc[i,1]==1 and sum(result_all.iloc[i,12:15])<=0:
        if sum(result_all.iloc[i,9:12])>=5000 and

np.savetxt('resultallxx.csv',np.c_[result_all.ID,result_all.Default],delimiter=',',header='ID,Default',comments='',fmt='%d')