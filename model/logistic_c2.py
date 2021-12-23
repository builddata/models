"""
Logistic回归：最基础的神经网络
一）数据预处理
搞清楚数据的形状、维度
将数据（例如图片）转化成向量（image to vector）方便处理
将数据标准化（standardize），这样更好训练
（二）构造各种辅助函数
激活函数（此处我们使用sigmoid函数）--activation function
参数初始化函数（用来初始化W和b）--initialization
传播函数（这里是用来求损失cost并对W、b求导，即dW、db）--propagate
优化函数（迭代更新W和b，来最小化cost）--optimize
预测函数（根据学习到的W和b来进行预测）--predict
（三）综合上面的辅助函数，结合成一个模型
可以直接输入训练集、预测集、超参数，然后给出模型参数和准确率

神经网络无非就是在Logistic regression的基础上，多了几个隐层，每层多了一些神经元，
卷积神经网络无非就是再多了几个特殊的filter，多了一些有特定功能的层，但是核心都是跟Logistic Regression一样的：

前向传播求损失，
反向传播求倒数；
不断迭代和更新，
调参预测准确度
"""
import numpy as np
#一）数据预处理
# 导入数据，“_orig”代表这里是原始数据，我们还要进一步处理才能使用：
import utils
from utils import load_dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
#由数据集获取一些基本参数，如训练样本数m，图片大小：
m_train = train_set_x_orig.shape[0]  #训练集大小209
m_test = test_set_x_orig.shape[0]    #测试集大小50
num_px = train_set_x_orig.shape[1]  #图片宽度64，大小是64×64
#将图片数据向量化（扁平化）：
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
#对数据进行标准化：
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
#-------------------------------------------------------------------------------------------
#（二）构造各种辅助函数
#1. 激活函数/sigmoid函数：
def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a
#2. 参数初始化函数（给参数都初始化为0）：
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b
#3.propagate函数：这里再次解释一下这个propagate，它包含了forward-propagate和backward-propagate，
#即正向传播和反向传播。正向传播求的是cost，反向传播是从cost的表达式倒推W和b的偏导数，当然我们会先求出Z的偏导数。这两个方向的传播也是神经网络的精髓。
def propagate(w, b, X, Y):
    """
    传参:
    w -- 权重, shape： (num_px * num_px * 3, 1)
    b -- 偏置项, 一个标量
    X -- 数据集，shape： (num_px * num_px * 3, m),m为样本数
    Y -- 真实标签，shape： (1,m)

    返回值:
    cost， dw ，db，后两者放在一个字典grads里
    """
    #获取样本数m：
    m = X.shape[1]

    # 前向传播 ：
    A = sigmoid(np.dot(w.T,X)+b)    #调用前面写的sigmoid函数
    #损失函数（Loss function），来衡量y'和y的差距：L(y',y) = -[y·log(y')+(1 - y)·log(1 - y')]
    #当y=1时，L(y',y)=-log(y')，要使L最小，则y'要最大，则y'=1；当y=0时，L(y',y)=-log(1-y')，要使L最小，则y'要最小，则y'=0.
    #代价函数：损失均值，J(W,b) = 1/m·Σmi=1L(y'(i),y(i))，是W和b的函数，学习的过程就是寻找W和b使得J(W,b)最小化的过程。求最小值的方法是用梯度下降法
    cost = -(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))/m

    # 反向传播：求导数，当J(θ)下降到无法下降时为止，即J(θ)对θ的导数为0时，即dw=0
    dZ = A-Y
    dw = (np.dot(X,dZ.T))/m  #np.dot()是numpy包的矩阵乘法，就是点乘
    db = (np.sum(dZ))/m

    #返回值：
    grads = {"dw": dw,
             "db": db}

    return grads, cost
#4.optimize函数：有了上面这些函数的加持，optimize函数就很好写了，就是在迭代中调用各个我们刚刚写的函数就是：

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    #定义一个costs数组，存放每若干次迭代后的cost，从而可以画图看看cost的变化趋势：
    costs = []
    #进行迭代：
    for i in range(num_iterations):
        # 用propagate计算出每次迭代后的cost和梯度：
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]

        # 用上面得到的梯度来更新参数：
        w = w - learning_rate*dw
        b = b - learning_rate*db

        # 每100次迭代，保存一个cost看看：
        if i % 100 == 0:
            costs.append(cost)

        # 这个可以不在意，我们可以每100次把cost打印出来看看，从而随时掌握模型的进展：
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    #迭代完毕，将最终的各个参数放进字典，并返回：
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs
#5.predict函数：预测就很简单了，我们已经学到了参数W和b，那么让我们的数据经过配备这些参数的模型就可得到预测值。

def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))

    A = sigmoid(np.dot(w.T,X)+b)
    for  i in range(m):
        if A[0,i]>0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction
#（三）结合起来，搭建模型！
def logistic_model(X_train,Y_train,X_test,Y_test,learning_rate=0.1,num_iterations=2000,print_cost=False):
    #获特征维度，初始化参数：
    dim = X_train.shape[0]
    W,b = initialize_with_zeros(dim)

    #梯度下降，迭代求出模型参数：
    params,grads,costs = optimize(W,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    W = params['w']
    b = params['b']

    #用学得的参数进行预测：
    prediction_train = predict(W,b,X_test)
    prediction_test = predict(W,b,X_train)

    #计算准确率，分别在训练集和测试集上：
    accuracy_train = 1 - np.mean(np.abs(prediction_train - Y_train))
    accuracy_test = 1 - np.mean(np.abs(prediction_test - Y_test))
    print("Accuracy on train set:",accuracy_train )
    print("Accuracy on test set:",accuracy_test )

   #为了便于分析和检查，我们把得到的所有参数、超参数都存进一个字典返回出来：
    d = {"costs": costs,
         "Y_prediction_test": prediction_test ,
         "Y_prediction_train" : prediction_train ,
         "w" : W,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations,
         "train_acy":accuracy_train,
         "test_acy":accuracy_test
        }
    return d

d = logistic_model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)