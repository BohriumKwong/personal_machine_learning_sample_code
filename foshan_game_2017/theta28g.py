#coding=utf-8
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
import matplotlib.pylab as plt
from sklearn.preprocessing import Imputer
import operator
import time

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

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
    start_time = time.time()

    params={
    'booster':'gbtree',
    'objective': 'multi:softmax', #二分类的问题
    'num_class':2, # 类别数，与 multisoftmax 并用
    'eval_metric': 'merror',
    'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth':8, # 构建树的深度，越大越容易过拟合
    'lambda':24,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    #'max_delta_step':6,
    'subsample':0.8, # 随机采样训练样本
    'colsample_bytree':0.8, # 生成树时进行的列采样
    'min_child_weight':2,
    'scale_pos_weight' : 1,#因为类别不平衡
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.04, # 如同学习率
    'seed':1500,
    'nthread':8,# cpu 线程数
    #'eval_metric': 'auc'
    }

    plst = list(params.items())
    num_rounds = 10000 # 迭代次数
    train = pd.read_csv("./Train/Train/newtrainXX.csv")
    train.set_index('ID', inplace=True)
    train=train[train['BILL_AMT1']+train['BILL_AMT2']+train['BILL_AMT3']+train['BILL_AMT4']+train['BILL_AMT5']
    +train['BILL_AMT6']+train['PAY_AMT1']+train['PAY_AMT2']+train['PAY_AMT3']+train['PAY_AMT4']+train['PAY_AMT5']
    +train['PAY_AMT6'] -train['Default'] !=-1]
    #print(len(train))
    # train['PAY_B']= (train['BILL_AMT2'] <= train['PAY_AMT1']).astype(int)+(train['BILL_AMT3'] <= train['PAY_AMT2']).astype(int)\
    #                 +(train['BILL_AMT4'] <= train['PAY_AMT3']).astype(int)+(train['BILL_AMT5'] <= train['PAY_AMT4']).astype(int)\
    #                 +(train['BILL_AMT6'] <= train['PAY_AMT5']).astype(int)
    bins=[20,25,30,35,40,50,60,70,80]
    train['AGE']=pd.cut(train['AGE'],bins,labels=[20,25,30,35,40,50,60,70]).astype("int")
    bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,100]
    train['c00']=(train['BILL_AMT1'] - train['PAY_AMT6'])/train['CRED_LIMIT']
    train['c00']=pd.cut(train['c00'],bins,labels=[0,1,2,3,4,5,6,7,8,9,10]).astype("int")
    train['c01']=(train['BILL_AMT2'] - train['PAY_AMT1'])/train['CRED_LIMIT']
    train['c01']=pd.cut(train['c01'],bins,labels=[0,1,2,3,4,5,6,7,8,9,10]).astype("int")
    train['c02']=(train['BILL_AMT3'] - train['PAY_AMT2'])/train['CRED_LIMIT']
    train['c02']=pd.cut(train['c02'],bins,labels=[0,1,2,3,4,5,6,7,8,9,10]).astype("int")
    train['c03']=(train['BILL_AMT4'] - train['PAY_AMT3'])/train['CRED_LIMIT']
    train['c03']=pd.cut(train['c03'],bins,labels=[0,1,2,3,4,5,6,7,8,9,10]).astype("int")
    train['c04']=(train['BILL_AMT5'] - train['PAY_AMT4'])/train['CRED_LIMIT']
    train['c04']=pd.cut(train['c04'],bins,labels=[0,1,2,3,4,5,6,7,8,9,10]).astype("int")
    train['c05']=(train['BILL_AMT6'] - train['PAY_AMT5'])/train['CRED_LIMIT']
    train['c05']=pd.cut(train['c05'],bins,labels=[0,1,2,3,4,5,6,8,9,10,11]).astype("int")
    #########################################
    bins=[-100,0,0.01,0.2,0.5,0.7,100]
    train['cc1']=pd.cut(train['cc1'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    train['cc2']=pd.cut(train['cc2'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    train['cc3']=pd.cut(train['cc3'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    train['cc4']=pd.cut(train['cc4'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    train['cc5']=pd.cut(train['cc5'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    train['PAY6LIT']=pd.cut(train['PAY6LIT'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    ###########################################
    # train['P'] = round((train['BILL_AMT2'] - train['PAY_AMT1'] + train['BILL_AMT3'] - train['PAY_AMT2']+
    #                    train['BILL_AMT4'] - train['PAY_AMT3'] + train['BILL_AMT5'] - train['PAY_AMT4']+
    #                    train['BILL_AMT6'] - train['PAY_AMT5'] )/700)
    # bins=[200,400,600,700,900,1200,1500,1800,2200,3500,4800,7000]
    # train['P']=pd.cut(train['P'],bins,labels=[200,400,600,700,900,1200,1500,1800,2200,3500,5000]).astype("int")
    # train['PAY6LIT']=pd.cut(train['PAY6LIT'],bins,labels=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).astype("int")
    # train['P1']=round(train['PAY_AMT6']/100)
    train['CRED_LIMIT']=round(train['CRED_LIMIT']/1000)
    ##print(train.head(2))
    ##print(train.dtypes)
    ##print(pd.value_counts(train))
    # default_1 = train[['Default','SEX','EDUCATION','MARRIAGE','AGE']]#[(train.Default==1)]
    # ##print(default_1.head(2))
    # counts = default_1[u'SEX'].value_counts()
    # tsex = default_1.groupby(['Default','SEX'])
    # count_sex = tsex.size().unstack().fillna(0)
    # counte = default_1[u'EDUCATION'].value_counts()
    # tedu = default_1.groupby(['Default','EDUCATION'])
    # count_edu = tedu.size().unstack().fillna(0)
    # print(count_sex)
    # print(count_edu)
    #######
    train = train.drop(['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5',
                        'BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
                        'PAY_AMT5','PAY_AMT6','c01','c02','c03','PAY_6'],axis=1)
    #train = train.drop(['BILL_AMT2','PAY_AMT1'],axis=1)
    ####################
    #pd.qcut(train.AGE,7)
    train[np.isinf(train)] = 10
    train[np.isnan(train)] = 10
    train_xy,val = train_test_split(train, test_size = 0.3,random_state=0)
    # train_xy.to_csv('./train_xy.csv',index=True)
    # val.to_csv('val.csv',index=True)
    Y = train_xy.Default
    X = train_xy.drop(['Default'],axis=1)
    val_y = val.Default
    val_x = val.drop(['Default'],axis=1)
    xgb_val = xgb.DMatrix(val_x,label=val_y)
    xgb_train = xgb.DMatrix(X, label=Y)
    watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]
    #xgb_test = xgb.DMatrix(tests)

    # training model
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练

############################################
    model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)
    features = [x for x in X.columns ]
    ceate_feature_map(features)

    importance = model.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    #df.to_csv("./PersonalContentDate/feat_importance.csv", index=False)

    plt.figure()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.show()
    #model.save_model('H:/foshangame2017/model/foshangame.model')
    #model.dump_model('H:/foshangame2017/model/foshangame..txt')
    print ("best best_ntree_limit",model.best_ntree_limit)

    print ("跑到这里了model.predict")
    cost_time = time.time()-start_time
    print ("xgboost success!",'\n',"cost time:",cost_time,"(s)")
    # #xgboost交叉验证并输出rmse
    test_all = pd.read_csv("./Train/Train/newtestXX.csv")
    bins=[20,25,30,35,40,50,60,70,80]
    test_all['AGE']=pd.cut(test_all['AGE'],bins,labels=[20,25,30,35,40,50,60,70]).astype("int")
    bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,100]
    test_all['c00']=(test_all['BILL_AMT1'] - test_all['PAY_AMT6'])/test_all['CRED_LIMIT']
    test_all['c00']=pd.cut(test_all['c00'],bins,labels=[0,1,2,3,4,5,6,7,8,9,10]).astype("int")
    test_all['c01']=(test_all['BILL_AMT2'] - test_all['PAY_AMT1'])/test_all['CRED_LIMIT']
    test_all['c01']=pd.cut(test_all['c01'],bins,labels=[0,1,2,3,4,5,6,7,8,9,10]).astype("int")
    test_all['c02']=(test_all['BILL_AMT3'] - test_all['PAY_AMT2'])/test_all['CRED_LIMIT']
    test_all['c02']=pd.cut(test_all['c02'],bins,labels=[0,1,2,3,4,5,6,7,8,9,10]).astype("int")
    test_all['c03']=(test_all['BILL_AMT4'] - test_all['PAY_AMT3'])/test_all['CRED_LIMIT']
    test_all['c03']=pd.cut(test_all['c03'],bins,labels=[0,1,2,3,4,5,6,7,8,9,10]).astype("int")
    test_all['c04']=(test_all['BILL_AMT5'] - test_all['PAY_AMT4'])/test_all['CRED_LIMIT']
    test_all['c04']=pd.cut(test_all['c04'],bins,labels=[0,1,2,3,4,5,6,7,8,9,10]).astype("int")
    test_all['c05']=(test_all['BILL_AMT6'] - test_all['PAY_AMT5'])/test_all['CRED_LIMIT']
    test_all['c05']=pd.cut(test_all['c05'],bins,labels=[0,1,2,3,4,5,6,8,9,10,11]).astype("int")

    bins=[-100,0,0.01,0.2,0.5,0.7,100]
    test_all['cc1']=pd.cut(test_all['cc1'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    test_all['cc2']=pd.cut(test_all['cc2'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    test_all['cc3']=pd.cut(test_all['cc3'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    test_all['cc4']=pd.cut(test_all['cc4'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    test_all['cc5']=pd.cut(test_all['cc5'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")
    test_all['PAY6LIT']=pd.cut(test_all['PAY6LIT'],bins,labels=[0,0.01,0.2,0.5,0.7,1]).astype("float")

    test_all['CRED_LIMIT']=round(test_all['CRED_LIMIT']/1000)
    test_all = test_all.drop(['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5',
                        'BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
                        'PAY_AMT5','PAY_AMT6','c01','c02','c03','PAY_6'],axis=1)
    test_all[np.isinf(test_all)] = 10
    test_all[np.isnan(test_all)] = 10
    xgb_test = xgb.DMatrix(test_all.drop(['ID'],axis=1))
    preds = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
    np.savetxt('xgbxx.csv',np.c_[test_all.ID,preds],delimiter=',',header='ID,Default',comments='',fmt='%d')
    preds = model.predict(xgb_val,ntree_limit=model.best_ntree_limit)
    # val['xgbresult']= preds
    # val.to_csv("./Train/xgbXX.csv")
    F1 = performance(val_y.values,preds)
    print('F1分数为:',F1)
###########################
# Stopping. Best iteration: 2017-08-22 11:00
# [92]	train-merror:0.166286	val-merror:0.175704

# Stopping. Best iteration: 2017-08-22 11:10
# [88]	train-merror:0.166286	val-merror:0.175111

# Stopping. Best iteration:  2017-08-22 12:12
# [283]	train-merror:0.123302	val-merror:0.175556

# Stopping. Best iteration: 2017-08-22 14:30
# [161]	train-merror:0.112508	val-merror:0.176148