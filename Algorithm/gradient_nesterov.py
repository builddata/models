import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return x[0] * x[0] + 50 * x[1] * x[1]

def g(x):
    return np.array([2 * x[0], 100 * x[1]])

xi = np.linspace(-200, 200, 1000)
yi = np.linspace(-100, 100, 1000)
X, Y = np.meshgrid(xi, yi)
Z = X * X + 50 * Y * Y

def contour(X, Y, Z, arr=None):
    plt.figure(figsize=(15, 7))
    xx = X.flatten()
    yy = Y.flatten()
    zz = Z.flatten()
    plt.contour(X, Y, Z, colors='black')
    plt.plot(0, 0, marker='*')
    if arr is not None:
        arr = np.array(arr)
        for i in range(len(arr) - 1):
            plt.plot(arr[i:i + 2, 0], arr[i:i + 2, 1])


# contour(X, Y, Z)
#均值归一化的目的是使数据都缩放到一个范围内，便于使用梯度下降算法
# 归一化feature
def featureNormaliza(X):
    X_norm = np.array(X)            #将X转化为numpy数组对象，才可以进行矩阵的运算
    #定义所需变量
    # mu = np.zeros((1,X.shape[1]))
    # sigma = np.zeros((1,X.shape[1]))
    mu = np.mean(X_norm,0)          # 求每一列的平均值（0指定为列，1代表行）
    sigma = np.std(X_norm,0)        # 求每一列的标准差
    for i in range(X.shape[1]):     # 遍历列
        X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i]  # 归一化
    return X_norm,mu,sigma

def nesterov(x_start, step, g, discount=0.7):  # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(50):
        x_future = x - step * discount * pre_grad
        grad = g(x_future)
        pre_grad = pre_grad * discount + grad
        x -= pre_grad * step
        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot


res, x_arr = nesterov([150, 75], 0.012, g)
contour(X, Y, Z, x_arr)
plt.show()