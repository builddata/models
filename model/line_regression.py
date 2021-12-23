# -*- coding: utf-8 -*-
"""线性回归-单一变量线性回归
适用于low dimension，而且每一维之间都没有共线性
优点:直接,快速,可解释性好。
缺点:需要严格的假设,需处理异常值，存在共线性，自相关，异方差的问题
模型参数的估计很不稳定， 模型中输入数据的微小差异都可能导致参数估计的很多差异
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#设置字体的更多属性
font={
      'family':'SimHei',
      'weight':'bold',
      'size' : '16'
      }
plt.rc('font',**font)
#解决坐标轴负轴的符号显示的问题
plt.rc('axes',unicode_minus=False)
#损耗函数算法代码，theta.T转置，矩阵计算
def computeCost(X,y,theta):

    m=len(y)
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * m)
#批量梯度下降算法
# 该函数通过执行梯度下降算法次数来更新theta值，每次迭代次数跟学习率有关
#   函数参数说明:
#   X :代表特征/输入变量
#   y:代表目标变量/输出变量
#   theta：线性回归模型的两个系数值（h(x)=theta(1)+theta(2)*x）
#   alpha：学习率
#   iters：迭代次数
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost
if __name__ == '__main__':
#读取训练数据集中的数据
#%文件ex1data1.txt包含了我们的线性回归问题的数据集。
#第一列是城市的人口（单位100000），第二列是城市里的一辆食品卡车的利润。
#利润的负值表示损失
    train_data=pd.read_csv('ex1data1.txt',names=['Population','Profit'])

#将训练集中的数据在图中显示
    train_data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
    plt.show()

#我们在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度。
    train_data.insert(0,'Ones',1)
    X=train_data.iloc[:,[0,1]]#X是所有行，去掉最后一列
    y=train_data.iloc[:,2]#y是所有行，最后一列
    #代价函数是矩阵计算，所以需要将X,y,theta转变为矩阵
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    y=y.T#转置
    theta = np.matrix(np.array([0,0]))
    print(X.shape, theta.shape, y.shape)
    #初始化一些附加变量 - 学习速率α和要执行的迭代次数。
    iters = 5000#迭代次数设置为1500次
    alpha = 0.01#学习率设置为0.01.
    computeCost(X, y, theta)
    g, cost = gradientDescent(X, y, theta, alpha, iters)#g是求出的最佳theat值，cost是所有迭代次数的代价函数求出的值
    print('求出的最佳theat值：',g)
    #将线性回归函数画出
    x = np.linspace(train_data.Population.min(), train_data.Population.max(), 100)
    f = g[0, 0] + (g[0, 1] * x)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, f, 'r', label='预测函数')
    ax.scatter(train_data.Population, train_data.Profit, label='训练数据')
    ax.legend(loc=2)
    ax.set_xlabel('人口')
    ax.set_ylabel('利润')
    ax.set_title('预测利润和人口数量')
    plt.show()
    # 预测人口规模为3.5万和7万的利润值
    predict1 = g[0,0]*1+(g[0, 1] * 3.5)
    print('当人口为35,000时,我们预测利润为',predict1*10000);
    predict2 = g[0,0]*1+(g[0, 1] * 7)
    print('当人口为70,000时,我们预测利润为',predict2*10000);
    #由于梯度方程式函数也在每个训练迭代中输出一个代价的向量，
    #所以我们也可以绘制。 请注意，代价总是降低 - 这是凸优化问题的一个例子。
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('输出代价')
    ax.set_title('误差和训练状态')
    plt.show()

#---------------------------------------------------------------------------------
"""线性回归-多元线性回归
称为多变量线性回归，我们认为各个特征和预测值之间是简单的加权求和关系
"""
import numpy as np  
from sklearn import linear_model  
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split #导入数据分割器
from sklearn.preprocessing import StandardScaler

xx, yy = np.meshgrid(np.linspace(0,10,10), np.linspace(0,100,10)) 
zz = 1.0 * xx + 3.5 * yy + np.random.randint(0,100,(10,10)) 
#如果差异较大，需要对特征及目标值标准差处理y=(yi-u)/σ
print(np.max(zz),np.min(zz),np.mean(zz))
ss_x=StandardScaler()
ss_y=StandardScaler()
xx=ss_x.fit_transform(xx)
yy=ss_x.fit_transform(yy)
zz=ss_x.fit_transform(zz)
# 构建成特征、值的形式 
X, y = np.column_stack((xx.flatten(),yy.flatten())), zz.flatten() 
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)

# 建立线性回归模型 
lr = linear_model.LinearRegression() 
# 拟合 
lr.fit(x_train, y_train) 
# 不难得到平面的系数、截距 
a, b = lr.coef_, lr.intercept_ 

#对测试数据进行回顾预测
lr_y_predict=lr.predict(x_test)
#评价
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print (r2_score(y_test,lr_y_predict))
# print (mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
# print (mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
# 给出待预测的一个特征 
x = np.array([[5.8, 78.3]])  
# 方式1：根据线性方程计算待预测的特征x对应的值z（注意：np.sum） 
print(np.sum(a * x) + b)  
# 方式2：根据predict方法预测的值z 
print(lr.predict(x))  
# 画图 
#----------------------------------------------------------------------------
import numpy as np  
from sklearn import linear_model  
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt  
xx, yy = np.meshgrid(np.linspace(0,10,10), np.linspace(0,100,10)) 
zz = 1.0 * xx + 3.5 * yy + np.random.randint(0,100,(10,10))  
# 构建成特征、值的形式 
X, Z = np.column_stack((xx.flatten(),yy.flatten())), zz.flatten() 
  
# 建立线性回归模型 
regr = linear_model.LinearRegression() 
  
# 拟合 
regr.fit(X, Z) 
# 不难得到平面的系数、截距 
a, b = regr.coef_, regr.intercept_  
# 给出待预测的一个特征 
x = np.array([[5.8, 78.3]])  
# 方式1：根据线性方程计算待预测的特征x对应的值z（注意：np.sum） 
print(np.sum(a * x) + b)  
# 方式2：根据predict方法预测的值z 
print(regr.predict(x))  
# 画图 
fig = plt.figure() 
ax = fig.gca(projection='3d')  
# 1.画出真实的点 
ax.scatter(xx, yy, zz) 
# 2.画出拟合的平面 
ax.plot_wireframe(xx, yy, regr.predict(X).reshape(10,10)) 
ax.plot_surface(xx, yy, regr.predict(X).reshape(10,10), alpha=0.3) 
 
plt.show()