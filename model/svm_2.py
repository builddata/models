"""
参数：
l  C：C-SVC的惩罚参数C?默认值是1.0
C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
l  kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
  　　0 – 线性：u'v
 　　 1 – 多项式：(gamma*u'*v + coef0)^degree
  　　2 – RBF函数：exp(-gamma|u-v|^2)
  　　3 –sigmoid：tanh(gamma*u'*v + coef0)
l  degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
l  gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
l  coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
l  probability ：是否采用概率估计？.默认为False
l  shrinking ：是否采用shrinking heuristic方法，默认为true
l  tol ：停止训练的误差值大小，默认为1e-3
l  cache_size ：核函数cache缓存大小，默认为200
l  class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
l  verbose ：允许冗余输出？
l  max_iter ：最大迭代次数。-1为无限制。
l  decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3
kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
　　 kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
　　decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
　　decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
主要调节的参数有：C、kernel、degree、gamma、coef0。
"""
