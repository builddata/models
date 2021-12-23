"""
LDA:
LDA在模式识别领域（比如人脸识别，舰艇识别等图形图像识别领域）中有非常广泛的应用
线性判别模型（LDA）在模式识别领域（比如人脸识别等图形图像识别领域）中有非常广泛的应用。LDA是一种监督学习的降维技术，
也就是说它的数据集的每个样本是有类别输出的。这点和PCA不同。PCA是不考虑样本类别输出的无监督降维技术。LDA的思想可以用一句话概括，
就是“投影后类内方差最小，类间方差最大”。我们要将数据在低维度上进行投影，投影后希望每一种类别数据的投影点尽可能的接近，
而不同类别的数据的类别中心之间的距离尽可能的大。即：将数据投影到维度更低的空间中，使得投影后的点，会形成按类别区分，
一簇一簇的情况，相同类别的点，将会在投影后的空间中更接近方法。
LDA算法的一个目标是使得不同类别之间的距离越远越好，同一类别之中的距离越近越好
在以后的实战中需要注意LDA在非线性可分数据上的谨慎使用
LDA算法的主要优点：
    1.在降维过程中可以使用类别的先验知识经验，而像PCA这样的无监督学习则无法使用类别先验知识；
    2.LDA在样本分类信息依赖均值而不是方差的时候，比PCA之类的算法较优。

LDA算法的主要缺点：
    1.LDA不适合对非高斯分布样本进行降维，PCA也有这个问题
    2.LDA降维最多降到类别数 k-1 的维数，如果我们降维的维度大于 k-1，则不能使用 LDA。当然目前有一些LDA的进化版算法可以绕过这个问题
    3.LDA在样本分类信息依赖方差而不是均值的时候，降维效果不好
    4.LDA可能过度拟合数据,
"""
#基于LDA手写数字分类实践
# 导入手写数据集 MNIST
from sklearn.datasets import load_digits
# 导入训练集分割方法
import matplotlib.pyplot as plt
# 导入三维显示工具
from mpl_toolkits.mplot3d import Axes3D
# 导入demo数据制作方法
from sklearn.datasets.samples_generator import make_classification
from sklearn.model_selection import train_test_split
# 导入LDA模型
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 导入预测指标计算函数和混淆矩阵计算函数
from sklearn.metrics import classification_report, confusion_matrix
# 导入绘图包
import seaborn as sns
import matplotlib
# 导入MNIST数据集
mnist = load_digits()

# 查看数据集信息
print('The Mnist dataeset:\n',mnist)

# 分割数据为训练集和测试集
x, test_x, y, test_y = train_test_split(mnist.data, mnist.target, test_size=0.1, random_state=2)
## 输出示例图像
images = range(0,9)

plt.figure(dpi=100)
for i in images:
    plt.subplot(330 + 1 + i)
    plt.imshow(x[i].reshape(8, 8), cmap = matplotlib.cm.binary,interpolation="nearest")
# show the plot

plt.show()
# 建立 LDA 模型
m_lda = LinearDiscriminantAnalysis()
# 进行模型训练
m_lda.fit(x, y)
# 进行模型预测
x_new = m_lda.transform(x)
# 可视化预测数据
plt.scatter(x_new[:, 0], x_new[:, 1], marker='o', c=y)
plt.title('MNIST with LDA Model')
plt.show()
# 进行测试集数据的类别预测
y_test_pred = m_lda.predict(test_x)
print("测试集的真实标签:\n", test_y)
print("测试集的预测标签:\n", y_test_pred)
# 进行预测结果指标统计 统计每一类别的预测准确率、召回率、F1分数
print(classification_report(test_y, y_test_pred))
# 计算混淆矩阵
C2 = confusion_matrix(test_y, y_test_pred)
# 打混淆矩阵
print(C2)

# 将混淆矩阵以热力图的防线显示
sns.set()
f, ax = plt.subplots()
# 画热力图
sns.heatmap(C2, cmap="YlGnBu_r", annot=True, ax=ax)
# 标题
ax.set_title('confusion matrix')
# x轴为预测类别
ax.set_xlabel('predict')
# y轴实际类别
ax.set_ylabel('true')
plt.show()
