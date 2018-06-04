#coding=utf-8
import numpy as np
from numpy import nan as NA
import pandas as pd

train_basic = pd.read_csv("./Train/Train/basicInfo_train.csv")
train_history = pd.read_csv("./Train/Train/history_train.csv")
# train_history.PAY_2[train_history.PAY_2==-2 and train_history.BILL_AMT2>train_history.PAY_AMT1]=0
# train_history.PAY_2[train_history.PAY_2==-2 and train_history.BILL_AMT2<=train_history.PAY_AMT1]=-1
# train_history.PAY_3[train_history.PAY_3==-2 and train_history.BILL_AMT3>train_history.PAY_AMT2]=0
# train_history.PAY_3[train_history.PAY_3==-2 and train_history.BILL_AMT3<=train_history.PAY_AMT2]=-1
# train_history.PAY_4[train_history.PAY_4==-2 and train_history.BILL_AMT4>train_history.PAY_AMT13]=0
# train_history.PAY_4[train_history.PAY_4==-2 and train_history.BILL_AMT4<=train_history.PAY_AMT3]=-1
# train_history.PAY_2[train_history.PAY_5==-2 and train_history.BILL_AMT5>train_history.PAY_AMT4]=0
# train_history.PAY_2[train_history.PAY_5==-2 and train_history.BILL_AMT5<=train_history.PAY_AMT4]=-1
# train_history.PAY_6[train_history.PAY_6==-2 and train_history.BILL_AMT6>train_history.PAY_AMT5]=0
# train_history.PAY_6[train_history.PAY_6==-2 and train_history.BILL_AMT6<=train_history.PAY_AMT5]=-1
for i in range(len(train_history)):
    if train_history.iloc[i,2]==-2:
        if train_history.iloc[i,8]<=train_history.iloc[i,13] :
            train_history.iloc[i,2]=-1
        else:
            train_history.iloc[i,2]=0
    elif train_history.iloc[i,3]==-2:
        if train_history.iloc[i,9]<=train_history.iloc[i,14]:
            train_history.iloc[i,3]=-1
        else:
            train_history.iloc[i,3]=0
    elif train_history.iloc[i,4]==-2:
        if train_history.iloc[i,10]<=train_history.iloc[i,15]:
            train_history.iloc[i,4]=-1
        else:
            train_history.iloc[i,4]=0
    elif train_history.iloc[i,5]==-2:
        if train_history.iloc[i,11]<=train_history.iloc[i,16]:
            train_history.iloc[i,5]=-1
        else:
            train_history.iloc[i,5]=0
    elif train_history.iloc[i,6]==-2:
        if train_history.iloc[i,12]<=train_history.iloc[i,17]:
            train_history.iloc[i,6]=-1
        else:
            train_history.iloc[i,6]=0
train_default = pd.read_csv("/Train/Train/default_train.csv")
train = pd.merge(pd.merge(train_basic,train_history,on ='ID'),train_default,on ='ID')

train.set_index('ID', inplace=True)
train['PAY_B']= (train['BILL_AMT2'] > train['PAY_AMT1']).astype(int)+(train['BILL_AMT3'] > train['PAY_AMT2']).astype(int)\
                    +(train['BILL_AMT4'] > train['PAY_AMT3']).astype(int)+(train['BILL_AMT5'] > train['PAY_AMT4']).astype(int)\
                    +(train['BILL_AMT6'] > train['PAY_AMT5']).astype(int)
train['cc1']=(train['PAY_AMT1'] /train['BILL_AMT2'])#.fillna(0.001)
train['cc2']=(train['PAY_AMT2'] /train['BILL_AMT3'])#.fillna(0.001)
train['cc3']=(train['PAY_AMT3'] /train['BILL_AMT4'])#.fillna(0.001)
train['cc4']=(train['PAY_AMT4'] /train['BILL_AMT5'])#.fillna(0.001)
train['cc5']=(train['PAY_AMT5'] /train['BILL_AMT6'])#.fillna(0.001)
train['PAY6LIT']=train['PAY_AMT6']/(train['PAY_AMT1']+train['PAY_AMT2']+train['PAY_AMT3']+train['PAY_AMT4']+train['PAY_AMT5'])*5

train[np.isinf(train)] = 2
train[np.isnan(train)] = 2
#train_history[train_history<0]=0
train.to_csv("/Train/Train/newtrainXX.csv")