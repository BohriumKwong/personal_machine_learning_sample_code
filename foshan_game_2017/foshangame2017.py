#coding=utf-8
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import matplotlib.pylab as plt
import time
start_time = time.time()

params={
'booster':'gbtree',
'objective': 'multi:softmax', #二分类的问题
'num_class':2, # 类别数，与 multisoftmax 并用
'eval_metric': 'merror',
'gamma':0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':10, # 构建树的深度，越大越容易过拟合
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'max_delta_step':3,
'subsample':0.8, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':0.4,
'scale_pos_weight' : 1,#因为类别不平衡
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.1, # 如同学习率
'seed':1000,
'nthread':8,# cpu 线程数
#'eval_metric': 'auc'
}

plst = list(params.items())
num_rounds = 5000 # 迭代次数
train_basic = pd.read_csv("./Train/Train/basicInfo_train.csv")
train_history = pd.read_csv("./Train/Train/history_train.csv")
train_default = pd.read_csv("./Train/Train/default_train.csv")
train = pd.merge(pd.merge(train_basic,train_history,on ='ID'),train_default,on ='ID')
train.set_index('ID', inplace = True)
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

train_xy,val = train_test_split(train, test_size = 0.3,random_state=0)

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
model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)

model.save_model('H:/foshangame2017/model/foshangame.model')
model.dump_model('H:/foshangame2017/model/foshangame..txt')
print ("best best_ntree_limit",model.best_ntree_limit)

print ("跑到这里了model.predict")
cost_time = time.time()-start_time
print ("xgboost success!",'\n',"cost time:",cost_time,"(s)")
#xgboost交叉验证并输出rmse
test_basic = pd.read_csv("./PersonalContentDate/basicInfo_test.csv")
test_history = pd.read_csv("./PersonalContentDate/history_test.csv")
test_all = pd.merge(test_basic,test_history,on ='ID')
#test_x = test_all.drop(['ID'],axis=1)
xgb_test = xgb.DMatrix(test_all.drop(['ID'],axis=1))
preds = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
np.savetxt('./submission.csv',np.c_[test_all.ID,preds],delimiter=',',header='ID,Default',comments='',fmt='%d')

# Stopping. Best iteration: 2017-08-22 11:00
# [92]	train-merror:0.166286	val-merror:0.175704

# Stopping. Best iteration: 2017-08-22 11:10
# [88]	train-merror:0.166286	val-merror:0.175111

# Stopping. Best iteration:  2017-08-22 12:12
# [283]	train-merror:0.123302	val-merror:0.175556

# Stopping. Best iteration: 2017-08-22 14:30
# [161]	train-merror:0.112508	val-merror:0.176148