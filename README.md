#个人python 机器学习代码整理





---



## Overview





本项目仅作为个人代码备忘录的用途，不能保证会定期维护更新。截止到2018-6-4，主要内容有一下部分:

 1. 比赛代码
 2. python预处理技巧(pandas,numpy等）
 3. 某些课程的作业代码心得

---



##  比赛代码范例



主要涵盖了一些比赛代码，针对同一个题目，用不同的预处理方法和模型，以及一些特征工程的技巧。



### 一些代码

```python

# 巧用pandas的groupby方法遍历非数值型离散变量的取值，并对其进行数值赋值，自动
# 生成map，通过映射map的方法实现格式换转，便于调用xgboos和其他基于树方法的模型
# 该方法不是最优雅和简洁的，但应该是最直观朴实的，而且训练样本中的变量map形成之
# 后可以直接对测试样本进行映射，不需要考虑分布问题

if __name__ == '__main__':
		train_origin = pd.read_excel(r"train.xlsx",sheetname='train')
	    train_origin.set_index('ID', inplace=True)
	    print (train_origin.shape)
	    train = train_origin
	    #Device_Type
	    Device_Type_flag = {}
	    i = 0
	    for name,group in train.groupby('Device_Type'):
	        Device_Type_flag[name] = 1 + i * 0.001
	        i = i + 1
	    print('Device_Type has been transformed, the len of dict is :',len(Device_Type_flag))
	    train.Device_Type = train.Device_Type.map(Device_Type_flag)
	    #Employer_Name
	    train.Employer_Name = category = pd.Categorical(train.Employer_Name).codes
	#     print(train.groupby('Employer_Name').size())
	    Employer_Name_flag = {}
	    i = 0
	    for name,group in train.groupby('Employer_Name'):
	        Employer_Name_flag[name] = 1 + i * 0.0001
	        i = i + 1
	    print('Employer_Name has been transformed, the len of dict is :',len(Employer_Name_flag))
	#     print(Employer_Name_flag)
	    train.Employer_Name = train.Employer_Name.map(Employer_Name_flag)
	#     print (train.Employer_Name)
 
	    #Salary_Account
	    Salary_Account_flag = {}
	    i = 0
	    for name,group in train.groupby('Salary_Account'):
	        Salary_Account_flag[name] = 1 + i * 0.001
	        i = i + 1
	    print('Salary_Account has been transformed, the len of dict is :',len(Salary_Account_flag))
	    train.Salary_Account = train.Salary_Account.map(Salary_Account_flag)
	     
	    #City
	    City_flag = {}
	    i = 0
	    for name,group in train.groupby('City'):
	        City_flag[name] = 1 + i * 0.001
	        i = i + 1
	    print('City has been transformed, the len of dict is :',len(City_flag))
	    train.City = train.City.map(City_flag)
	    #Var1
	    Var1_flag = {}
	    i = 0
	    for name,group in train.groupby('Var1'):
	        Var1_flag[name] = 1 + i * 0.001
	        i = i + 1
	    print('Var1 has been transformed, the len of dict is :',len(Var1_flag)) 
	    train.Var1 = train.Var1.map(Var1_flag)
	    #Var2
	    Var2_flag = {}
	    i = 0
	    for name,group in train.groupby('Var2'):
	        Var2_flag[name] = 1 + i * 0.01
	        i = i + 1
	    print('Var2 has been transformed, the len of dict is :',len(Var2_flag)) 
	    train.Var2 = train.Var2.map(Var2_flag)
	    #Source
	    Source_flag = {}
	    i = 0
	    for name,group in train.groupby('Source'):
	        Source_flag[name] = 1 + i * 0.001
	        i = i + 1
	    print('Source has been transformed, the len of dict is :',len(Source_flag))
	    train.Source = train.Source.map(Source_flag)
	    train.duplicated()
	    train[(True^train['Disbursed'].apply(lambda x: str(x).isspace()))]
	    print (train.shape)
	    train['Gender'] = (train['Gender']=='male').astype(int)
	    train['Mobile_Verified'] = (train['Mobile_Verified']=='Y').astype(int)
	    train['Filled_Form'] = (train['Filled_Form']=='Y').astype(int)
	    from datetime import datetime
	#     print(train.head(2))
	    train['Age']=[2018 - i.year for i in train.DOB]
	    print(train.Age.head(5))



```













##Contact me：



<bohrium.kwong@gmail.com>







