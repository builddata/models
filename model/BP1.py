"""
BP
BP神经网络具有以下优点：
1) 非线性映射能力：BP神经网络实质上实现了一个从输入到输出的映射功能，数学理论证明三层的神经网络就能够以任意精度逼近任何非线性连续函数。
这使得其特别适合于求解内部机制复杂的问题，即BP神经网络具有较强的非线性映射能力。
2) 自学习和自适应能力：BP神经网络在训练时，能够通过学习自动提取输入、输出数据间的“合理规则”，并自适应地将学习内容记忆于网络的权值中。
即BP神经网络具有高度自学习和自适应的能力。
3) 泛化能力：所谓泛化能力是指在设计模式分类器时，即要考虑网络在保证对所需分类对象进行正确分类，还要关心网络在经过训练后，
能否对未见过的模式或有噪声污染的模式，进行正确的分类。也即BP神经网络具有将学习成果应用于新知识的能力。

BP神经网络具有以下缺点点：
1) 局部极小化问题：从数学角度看，传统的 BP神经网络为一种局部搜索的优化方法，它要解决的是一个复杂非线性化问题，
网络的权值是通过沿局部改善的方向逐渐进行调整的，这样会使算法陷入局部极值，权值收敛到局部极小点，从而导致网络训练失败。
加上BP神经网络对初始网络权重非常敏感，以不同的权重初始化网络，其往往会收敛于不同的局部极小，这也是每次训练得到不同结果的根本原因。
2) BP 神经网络算法的收敛速度慢：由于BP神经网络算法本质上为梯度下降法，它所要优化的目标函数是非常复杂的，因此，必然会出现“锯齿形现象”，
这使得BP算法低效；又由于优化的目标函数很复杂，它必然会在神经元输出接近0或1的情况下，出现一些平坦区，在这些区域内，权值误差改变很小，
使训练过程几乎停顿；BP神经网络模型中，为了使网络执行BP算法，不能使用传统的一维搜索法求每次迭代的步长，而必须把步长的更新规则预先赋予网络，
这种方法也会引起算法低效。以上种种，导致了BP神经网络算法收敛速度慢的现象。

BP神经网络模型要点在于数据的前向传播和误差反向传播，来对参数进行更新，使得损失最小化。 误差反向传播算法简称反向传播算法（即BP算法）。
使用反向传播算法的多层感知器又称为BP神经网络。BP算法是一个迭代算法，它的基本思想为：
（1）先计算每一层的状态和激活值，直到最后一层（即信号是前向传播的）；
（2）计算每一层的误差，误差的计算过程是从最后一层向前推进的（这就是反向传播算法名字的由来）；
（3）更新参数（目标是误差变小）。迭代前面两个步骤，直到满足停止准则（比如相邻两次迭代的误差的差别很小）。
在这个过程，函数的导数链式法则求导很重要，需要手动推导BP神经网络模型的梯度反向传播过程，熟练掌握链式法则进行求导，对参数进行更新。
"""
#基于BP神经网络的乳腺癌分类实践
# 导入乳腺癌数据集
from sklearn.datasets import load_breast_cancer
# 导入BP模型
from sklearn.neural_network import MLPClassifier
# 导入训练集分割方法
from sklearn.model_selection import train_test_split
# 导入预测指标计算函数和混淆矩阵计算函数
from sklearn.metrics import classification_report, confusion_matrix
# 导入绘图包
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 导入乳腺癌数据集
cancer = load_breast_cancer()
# 查看数据集信息
print('breast_cancer数据集的长度为：',len(cancer))
print('breast_cancer数据集的类型为：',type(cancer))
# 分割数据为训练集和测试集
cancer_data = cancer['data']
print('cancer_data数据维度为：',cancer_data.shape)
cancer_target = cancer['target']
print('cancer_target标签维度为：',cancer_target.shape)
cancer_names = cancer['feature_names']
cancer_desc = cancer['DESCR']
#分为训练集与测试集
cancer_data_train,cancer_data_test = train_test_split(cancer_data,test_size=0.2,random_state=42)#训练集
cancer_target_train,cancer_target_test = train_test_split(cancer_target,test_size=0.2,random_state=42)#测试集

# 建立 BP 模型, 采用Adam优化器，relu非线性映射函数
BP = MLPClassifier(solver='adam',activation = 'relu',max_iter = 1000,alpha = 1e-3,hidden_layer_sizes = (64,32, 32),random_state = 1)
# 进行模型训练
BP.fit(cancer_data_train, cancer_target_train)
# 进行模型预测
predict_train_labels = BP.predict(cancer_data_train)
# 可视化真实数据
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)
ax.scatter(cancer_data_train[:, 0], cancer_data_train[:, 1], cancer_data_train[:, 2], marker='o', c=cancer_target_train)
plt.title('True Label Map')
plt.show()
# 可视化预测数据
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)
ax.scatter(cancer_data_train[:, 0], cancer_data_train[:, 1], cancer_data_train[:, 2], marker='o', c=predict_train_labels)
plt.title('Cancer with BP Model')
plt.show()
# 显示预测分数
print("预测准确率: {:.4f}".format(BP.score(cancer_data_test, cancer_target_test)))
# 进行测试集数据的类别预测
predict_test_labels = BP.predict(cancer_data_test)
print("测试集的真实标签:\n", cancer_target_test)
print("测试集的预测标签:\n", predict_test_labels)
# 进行预测结果指标统计 统计每一类别的预测准确率、召回率、F1分数
print(classification_report(cancer_target_test, predict_test_labels))
# 计算混淆矩阵
confusion_mat = confusion_matrix(cancer_target_test, predict_test_labels)
# 打混淆矩阵
print(confusion_mat)
# 将混淆矩阵以热力图的防线显示
sns.set()
figure, ax = plt.subplots()
# 画热力图
sns.heatmap(confusion_mat, cmap="YlGnBu_r", annot=True, ax=ax)
# 标题
ax.set_title('confusion matrix')
# x轴为预测类别
ax.set_xlabel('predict')
# y轴实际类别
ax.set_ylabel('true')
plt.show()