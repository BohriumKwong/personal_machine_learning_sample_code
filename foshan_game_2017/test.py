#coding=utf-8
import numpy as np
import pandas as pd

test_basic = pd.read_csv("./PersonalContentDate/basicInfo_test.csv")
test_history = pd.read_csv("./PersonalContentDate/history_test.csv")
test_all = pd.merge(test_basic,test_history,on ='ID')
#train.set_index('ID', inplace = True)
test = test_all.drop(['BILL_AMT1'],axis=1)
test['Col_MUL'] = test['BILL_AMT6'] - test['PAY_AMT4']
#print(test['ID'].values)
test['AGE']=pd.qcut(test.AGE,10)
test['c01']=(test['BILL_AMT2'] - test['PAY_AMT1'])/test['CRED_LIMIT']
bins=[-100,0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1,100]
test['c011']=pd.cut(test['c01'],bins)
#test['c011']=pd.cut(test['c01'],11,labels=[-1,0,1,2,3,4,5,6,7,8,9])
#test['BIZHI']=pd.qcut('c01',10)
#if (test['BILL_AMT2'] - test['PAY_AMT1'])/test.['CRED_LIMIT']
print(test.iloc[12,2:4])
# df = pd.DataFrame(np.random.randn(6,4), index=list('abcdef'), columns=list('ABCD'))
# print(df,'结果为:',df.iloc[5,:].C)
# print(len(df))
#print(test.dtypes)
#print('结果为:',df.iloc[3,:].C)
test_history.PAY_1[test_history.PAY_1==-1]=0.1
for i in range(len(test_history)):
   # if sum(test_history.iloc[i,8:13])==0 and 1==1:
    if test_history.iloc[i,1]==0.1:
        print(test_history.iloc[i,0:4])
        #8:13 BILL_AMT2~6
        #13:18 PAY_AMT1~5
        #print(test_history.iloc[i,11:17])

# a=df.iloc[5,:].C*7
# print(a)