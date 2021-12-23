# -*- coding: utf-8 -*-
"""
Naive Bayes
它会单独考量每一唯独特征被分类的条件概率，进而综合这些概率并对其所在的特征向量做出分类预测。
因此，朴素贝叶斯的基本数据假设是：各个维度上的特征被分类的条件概率之间是相互独立的。它经常被应用在文本分类中，包括互联网新闻的分类，垃圾邮件的筛选。
"""
from sklearn.datasets import fetch_20newsgroups  # 从sklearn.datasets里导入新闻数据抓取器 fetch_20newsgroups
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # 从sklearn.feature_extraction.text里导入文本特征向量化模块
from sklearn.naive_bayes import MultinomialNB     # 从sklean.naive_bayes里导入朴素贝叶斯模型
from sklearn.metrics import classification_report

#1.数据获取
news = fetch_20newsgroups(subset='all')
print (len(news.data))  # 输出数据的条数：18846

#2.数据预处理：训练集和测试集分割，文本特征向量化
X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33) # 随机采样25%的数据样本作为测试集
#print X_train[0]  #查看训练样本
#print y_train[0:100]  #查看标签

#文本特征向量化
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

#3.使用朴素贝叶斯进行训练
mnb = MultinomialNB()   # 使用默认配置初始化朴素贝叶斯
mnb.fit(X_train,y_train)    # 利用训练数据对模型参数进行估计
y_predict = mnb.predict(X_test)     # 对参数进行预测

#4.获取结果报告
print ('The Accuracy of Naive Bayes Classifier is:', mnb.score(X_test,y_test))
print (classification_report(y_test, y_predict, target_names = news.target_names))