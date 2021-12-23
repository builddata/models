# -*- coding: utf-8 -*-
"""
SVM_RBF
如果C很小，对误分类点的惩罚很低，因此选择一个具有较大间隔的决策边界是以牺牲更多的错误分类为代价的。
当C值较大时，支持向量机会尽量减少误分类样本的数量，因为惩罚会导致决策边界具有较小的间隔。对于所有错误分类的例子，惩罚是不一样的。
它与到决策边界的距离成正比。
Gamma是用于非线性支持向量机的超参数。最常用的非线性核函数之一是径向基函数(RBF)。RBF的Gamma参数控制单个训练点的影响距离。
gamma值较低表示相似半径较大，这会导致将更多的点组合在一起。对于gamma值较高的情况，点之间必须非常接近，才能将其视为同一组(或类)。因此，具有非常大gamma值的模型往往过拟合。

"""

#创建一个RBF内核的支持向量机模型
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
#导入支持向量机svm
from sklearn import svm
#定义一个函数用来画图
def make_meshgrid(x,y,h=.02):
    x_min,x_max = x.min() - 1,x.max() + 1
    y_min,y_max = y.min() - 1,y.max() + 1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    return xx,yy
#定义一个绘制等高线的函数
def plot_contours(ax,clf,xx,yy,**params):
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx,yy,Z,**params)
    return out
 
#使用酒的数据集
wine = load_wine()
#选取数据集的前两个特征
X = wine.data[:, :2]
y = wine.target
from sklearn.model_selection import train_test_split #导入数据分割器

x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)

C = 1.0 #svm的正则化参数
#LinearSVC使用的是平方hinge loss，SVC使用的是绝对值hinge loss
#LinearSVC使用的是One-vs-All（也成One-vs-Rest）的优化方法，而SVC使用的是One-vs-One
models = (svm.SVC(kernel='linear',C=C,max_iter=1000),svm.LinearSVC(C=C,dual=False,max_iter=1000),svm.SVC(kernel='rbf',gamma=0.7,C=C,max_iter=1000))
models = (clf.fit(X,y) for clf in models)
 
#设定图题
titles = ('SVC with linear kernel','LineatSVC (linear kernel)','SVC with RBF kernel')
 
#设定一个子图形的个数和排列方式
flg, sub = plt.subplots(1, 3)
plt.subplots_adjust(wspace=0.4,hspace=1)
#使用前面定义的函数进行画图
X0,X1, = X[:, 0],X[:, 1]
xx,yy = make_meshgrid(X0,X1)
 
for clf,title,ax in zip(models,titles,sub.flatten()):
    plot_contours(ax,clf,xx,yy,cmap=plt.cm.plasma,alpha=0.8)
    
    print ("train",title,clf.score(x_train, y_train))  # 精度
    print ("test",title,clf.score(x_test, y_test))
   
    ax.scatter(X0,X1,c=y,cmap=plt.cm.plasma,s=20,edgecolors='k')
    ax.set_xlim(xx.min(),xx.max())
    ax.set_ylim(yy.min(),yy.max())
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Featuer 1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
#将图型显示出来
plt.show()