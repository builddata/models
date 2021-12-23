"""
Xgboost
由于sklearn中没有集成Xgboost，所以才需要单独下载安装
，Xgboost是以“正则化提升（regularized boosting）” 技术而闻名。Xgboost在代价函数里加入了正则项，
用于控制模型的复杂度。正则项里包含了树的叶子节点个数，每个叶子节点上输出的score的L2模的平方和.
Xgboost工具支持并行。，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），
Xgboost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复使用这个结构，大大减小计算量
Xgboost先从顶到底建立所有可以建立的子树，再从底到顶反向机芯剪枝,这样不容易陷入局部最优解
Xgboost允许在每一轮Boosting迭代中使用交叉验证。因此可以方便的获得最优Boosting迭代次数，而GBM使用网格搜索，只能检测有限个值。
"""
#可以使用如下方式处理DMatrix中的缺失值

dtrain = xgb.DMatrix( data, label=label, missing = -999.0)
#当需要给样本设置权重时，可以用如下方式：
w = np.random.rand(5,1)
dtrain = xgb.DMatrix( data, label=label, missing = -999.0, weight=w)
#Xgboost使用key-value字典的方式存储参数
# xgboost模型
params = {
    'booster':'gbtree',
    'objective':'multi:softmax',   # 多分类问题
    'num_class':10,  # 类别数，与multi softmax并用
    'gamma':0.1,    # 用于控制是否后剪枝的参数，越大越保守，一般0.1 0.2的样子
    'max_depth':12,  # 构建树的深度，越大越容易过拟合
    'lambda':2,  # 控制模型复杂度的权重值的L2 正则化项参数，参数越大，模型越不容易过拟合
    'subsample':0.7, # 随机采样训练样本
    'colsample_bytree':3,# 这个参数默认为1，是每个叶子里面h的和至少是多少
    # 对于正负样本不均衡时的0-1分类而言，假设h在0.01附近，min_child_weight为1
    #意味着叶子节点中最少需要包含100个样本。这个参数非常影响结果，
    # 控制叶子节点中二阶导的和的最小值，该参数值越小，越容易过拟合
    'silent':0,  # 设置成1 则没有运行信息输入，最好是设置成0
    'eta':0.007,  # 如同学习率
    'seed':1000,
    'nthread':7}  #CPU线程数

# 通用参数
# booster [default=gbtree]
# 有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。缺省值为gbtree
# silent [default=0]
# 取0时表示打印出运行时信息，取1时表示以缄默方式运行，不打印运行时的信息。缺省值为0
# 建议取0，过程中的输出数据有助于理解模型以及调参。另外实际上我设置其为1也通常无法缄默运行。。
# nthread [default to maximum number of threads available if not set]
# XGBoost运行时的线程数。缺省值是当前系统可以获得的最大线程数
# 如果你希望以最大速度运行，建议不设置这个参数，模型将自动获得最大线程
# num_pbuffer [set automatically by xgboost, no need to be set by user]
# size of prediction buffer, normally set to number of training instances. The buffers are used to save the prediction results of last boosting step.
# num_feature [set automatically by xgboost, no need to be set by user]
# boosting过程中用到的特征维数，设置为特征个数。XGBoost会自动设置，不需要手工设置
#--------------------------------------------------------------------------------------
# tree booster参数
# # eta [default=0.3]
# # 为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。 eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3
# # 取值范围为：[0,1]
# # 通常最后设置eta为0.01~0.2
# # gamma [default=0]
# # minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, the more conservative the algorithm will be.
# # range: [0,∞]
# # 模型在默认情况下，对于一个节点的划分只有在其loss function 得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值
# # gamma值使得算法更conservation，且其值依赖于loss function ，在模型中应该进行调参。
# # max_depth [default=6]
# # 树的最大深度。缺省值为6
# # 取值范围为：[1,∞]
# # 指树的最大深度
# # 树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合
# # 建议通过交叉验证（xgb.cv ) 进行调参
# # 通常取值：3-10
# # min_child_weight [default=1]
# # 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
# # 取值范围为: [0,∞]
# # max_delta_step [default=0]
# # Maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update
# # 取值范围为：[0,∞]
# # 如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
# # 通常不需要设置这个值，但在使用logistics 回归时，若类别极度不平衡，则调整该参数可能有效果
# # subsample [default=1]
# # 用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
# # 取值范围为：(0,1]
# # colsample_bytree [default=1]
# # 在建立树时对特征随机采样的比例。缺省值为1
# # 取值范围：(0,1]
# # colsample_bylevel[default=1]
# # 决定每次节点划分时子样例的比例
# # 通常不使用，因为subsample和colsample_bytree已经可以起到相同的作用了
# # scale_pos_weight[default=0]
# # A value greater than 0 can be used in case of high class imbalance as it helps in faster convergence.
# # 大于0的取值可以处理类别不平衡的情况。帮助模型更快收敛
#--------------------------------------------------------------------------------------
# Linear Booster参数
# lambda [default=0]
# L2 正则的惩罚系数
# 用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
# alpha [default=0]
# L1 正则的惩罚系数
# 当数据维度极高时可以使用，使得算法运行更快。
# lambda_bias
# 在偏置上的L2正则。缺省值为0（在L1上没有偏置项的正则，因为L1时偏置不重要）
#--------------------------------------------------------------------------------
# 学习目标参数
#
# objective [ default=reg:linear ]
# 定义学习任务及相应的学习目标，可选的目标函数如下：
# “reg:linear” –线性回归。
# “reg:logistic” –逻辑回归。
# “binary:logistic” –二分类的逻辑回归问题，输出为概率。
# “binary:logitraw” –二分类的逻辑回归问题，输出的结果为wTx。
# “count:poisson” –计数问题的poisson回归，输出结果为poisson分布。
# 在poisson回归中，max_delta_step的缺省值为0.7。(used to safeguard optimization)
# “multi:softmax” –让XGBoost采用softmax目标函数处理多分类问题，同时需要设置参数num_class（类别个数）
# “multi:softprob” –和softmax一样，但是输出的是ndata * nclass的向量，可以将该向量reshape成ndata行nclass列的矩阵。每行数据表示样本所属于每个类别的概率。

#Xgboost基本方法和默认参数
xgboost.train(params,dtrain,num_boost_round=10,evals(),obj=None,
feval=None,maximize=False,early_stopping_rounds=None,evals_result=None,
verbose_eval=True,learning_rates=None,xgb_model=None)
# parms：这是一个字典，里面包含着训练中的参数关键字和对应的值，形式是parms = {'booster':'gbtree','eta':0.1}
#
# 　　dtrain：训练的数据
#
# 　　num_boost_round：这是指提升迭代的个数
#
# 　　evals：这是一个列表，用于对训练过程中进行评估列表中的元素。形式是evals = [(dtrain,'train'),(dval,'val')] 或者是 evals =[(dtrain,'train')] ，对于第一种情况，它使得我们可以在训练过程中观察验证集的效果。
#
# 　　obj ：自定义目的函数
#
# 　　feval：自定义评估函数
#
# 　　maximize：是否对评估函数进行最大化
#
# 　　early_stopping_rounds：早起停止次数，假设为100，验证集的误差迭代到一定程度在100次内不能再继续降低，就停止迭代。这要求evals里至少有一个元素，如果有多个，按照最后一个去执行。返回的是最后的迭代次数（不是最好的）。如果early_stopping_rounds存在，则模型会生成三个属性，bst.best_score ,bst.best_iteration和bst.best_ntree_limit
#
# 　　evals_result：字典，存储在watchlist中的元素的评估结果
#
# 　　verbose_eval（可以输入布尔型或者数值型）：也要求evals里至少有一个元素，如果为True，则对evals中元素的评估结果会输出在结果中；如果输入数字，假设为5，则每隔5个迭代输出一次。
#
# 　　learning_rates：每一次提升的学习率的列表
#
# 　　xgb_model：在训练之前用于加载的xgb_model

模型训练
　　有了参数列表和数据就可以训练模型了

num_round = 10
bst = xgb.train( plst, dtrain, num_round, evallist )
5，模型预测

# X_test类型可以是二维List，也可以是numpy的数组
dtest = DMatrix(X_test)
ans = model.predict(dtest)


xgb_model.get_booster().save_model('xgb.model')
tar = xgb.Booster(model_file='xgb.model')
x_test = xgb.DMatrix(x_test)
pre=tar.predict(x_test)
act=y_test
print(mean_squared_error(act, pre))
　　

6，保存模型
　　在训练完成之后可以将模型保存下来，也可以查看模型内部的结构

bst.save_model('test.model')
7加载模型
　　通过如下方式可以加载模型

bst = xgb.Booster({'nthread':4}) # init model
bst.load_model("model.bin")      # load data
#----------------------------------------------------------------------------------------------------------
#Xgboost使用sklearn接口的分类（推荐
from sklearn.datasets import load_iris

import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # 准确率
from xgboost.sklearn import XGBClassifier

clf = XGBClassifier(
    silent=0,  # 设置成1则没有运行信息输出，最好是设置为0，是否在运行升级时打印消息
    # nthread = 4  # CPU 线程数 默认最大
    learning_rate=0.3,  # 如同学习率
    min_child_weight=1,
    # 这个参数默认为1，是每个叶子里面h的和至少是多少，对正负样本不均衡时的0-1分类而言
    # 假设h在0.01附近，min_child_weight为1 意味着叶子节点中最少需要包含100个样本
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易过拟合
    max_depth=6,  # 构建树的深度，越大越容易过拟合
    gamma=0,  # 树的叶子节点上做进一步分区所需的最小损失减少，越大越保守，一般0.1 0.2这样子
    subsample=1,  # 随机采样训练样本，训练实例的子采样比
    max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计
    colsample_bytree=1,  # 生成树时进行的列采样
    reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合
    # reg_alpha=0, # L1正则项参数
    # scale_pos_weight =1 # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛，平衡正负权重
    # objective = 'multi:softmax', # 多分类问题，指定学习任务和响应的学习目标
    # num_class = 10,  # 类别数，多分类与multisoftmax并用
    n_estimators=100,  # 树的个数
    seed=1000,  # 随机种子
# 记载样本数据集
iris = load_iris()
X, y = iris.data, iris.target
# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123457)

# 算法参数
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'slient': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

plst = params.items()

# 生成数据集格式
dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 500
# xgboost模型训练
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print('accuarcy:%.2f%%' % (accuracy * 100))

# 显示重要特征
plot_importance(model)
plt.show()
# clf = XGBClassifier(
#     silent=0,  # 设置成1则没有运行信息输出，最好是设置为0，是否在运行升级时打印消息
#     # nthread = 4  # CPU 线程数 默认最大
#     learning_rate=0.3 , # 如同学习率
#     min_child_weight = 1,
#     # 这个参数默认为1，是每个叶子里面h的和至少是多少，对正负样本不均衡时的0-1分类而言
#     # 假设h在0.01附近，min_child_weight为1 意味着叶子节点中最少需要包含100个样本
#     # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易过拟合
#     max_depth=6, # 构建树的深度，越大越容易过拟合
#     gamma = 0,# 树的叶子节点上做进一步分区所需的最小损失减少，越大越保守，一般0.1 0.2这样子
#     subsample=1, # 随机采样训练样本，训练实例的子采样比
#     max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计
#     colsample_bytree=1, # 生成树时进行的列采样
#     reg_lambda=1, #控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合
#     # reg_alpha=0, # L1正则项参数
#     # scale_pos_weight =1 # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛，平衡正负权重
#     # objective = 'multi:softmax', # 多分类问题，指定学习任务和响应的学习目标
#     # num_class = 10,  # 类别数，多分类与multisoftmax并用
#     n_estimators=100,  # 树的个数
#     seed = 1000,  # 随机种子
#     # eval_metric ='auc'
#-----------------------------------------------------------
#基于Xgboost原生接口的回归
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

# 加载数据集,此数据集时做回归的
boston = load_boston()
X, y = boston.data, boston.target

# Xgboost训练过程
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 算法参数
params = {
    'booster': 'gbtree',
    'objective': 'reg:gamma',
    'gamma': 0.1,
    'max_depth': 5,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'slient': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 300
plst = params.items()
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)

# 显示重要特征
plot_importance(model)
plt.show()