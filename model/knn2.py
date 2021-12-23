# -*- coding: utf-8 -*-
"""
KNN
输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，
然后算法提取样本集中特征最相似数据（最近邻）的分类标签。
通常k是不大于20的整数。最后，选择k个最相似数据中出现次数最多的分类，作为新数据的分类。
"""


from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split #导入数据分割器
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#决策边界函数
def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
    # 设置标记点和颜色
    markers = ('s','x','o','^','v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
 
    # 绘制决策面
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    #绘制所有样本
    X_test,y_test=X[test_idx,:],y[test_idx]
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,color=cmap(idx),marker=markers[idx],label=cl)
    #高亮预测样本
    if test_idx:
        X_test,y_test =X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='', alpha=1.0,linewidths=1,marker='o',s=55,label='test set')
#数据导入
iris=datasets.load_iris()
X=iris.data[:,[2,3]]
y=iris.target
print (np.unique(y))
#训练集和测试集划分
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#标准化
sc=StandardScaler()
sc.fit(X_train)#计算样本的均值和标准差
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
#惰性学习-实例学习：KNN
knn=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')#闵可夫斯基距离
knn.fit(X_train_std,y_train)
#模型预测
y_pred=knn.predict(X_test_std)
print ('Accuracy:%.2f' %accuracy_score(y_test,y_pred))#准确率
#绘制决策边界
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=knn, test_idx=range(105,150))
plt.xlabel('petal length[standardized]')
plt.ylabel('petal width[standardized]')
plt.legend(loc='upper left')
plt.show()



