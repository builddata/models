import numpy as np
import matplotlib.pyplot as plt
xi = np.linspace(-2, 2, 10)
yi = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(xi, yi)
Z = 100 * (X ** 2 - Y ** 2) ** 2 + (X - 1) ** 2

def contour(X, Y, Z, arr=None):
    plt.figure(figsize=(15, 7))
    xx = X.flatten()
    yy = Y.flatten()
    zz = Z.flatten()
    plt.contour(X, Y, Z, colors='black')
    plt.plot(1, 1, marker='*')
    if arr is not None:
        arr = np.array(arr)
        for i in range(len(arr) - 1):
            plt.plot(arr[i:i + 2, 0], arr[i:i + 2, 1])
# 函数表达式
fun = lambda x: 100 * (x[0] ** 2 - x[1] ** 2) ** 2 + (x[0] - 1) ** 2

# 梯度向量
gfun = lambda x: np.array([400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1), -200 * (x[0] ** 2 - x[1])])

# Hessian矩阵
hess = lambda x: np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])

def dfp(fun, gfun, hess, x0):
    # 功能：用DFP算法求解无约束问题：min fun(x)
    # 输入：x0式初始点，fun,gfun，hess分别是目标函数和梯度,Hessian矩阵格式
    # 输出：x,val分别是近似最优点，最优解，k是迭代次数
    maxk = 1e5
    rho = 0.05
    sigma = 0.4
    epsilon = 1e-5  # 迭代停止条件
    k = 0
    n = np.shape(x0)[0]
    # 将Hessian矩阵初始化为单位矩阵
    Hk = np.linalg.inv(hess(x0))
    passing_dot = [x0.copy()]
    while k < maxk:
        gk = gfun(x0)
        if np.linalg.norm(gk) < epsilon:
            break
        dk = -1.0 * np.dot(Hk, gk)
        #         print dk
        m = 0;
        mk = 0
        while m < 20:  # 用Armijo搜索步长
            if fun(x0 + rho ** m * dk) < fun(x0) + sigma * rho ** m * np.dot(gk, dk):
                mk = m
                break
            m += 1
        # print mk
        # DFP校正
        x = x0 + rho ** mk * dk
        print("第" + str(k) + "次的迭代结果为：" + str(x))
        sk = x - x0
        yk = gfun(x) - gk
        passing_dot.append(x.copy())
        if np.dot(sk, yk) > 0:
            Hy = np.dot(Hk, yk)
            sy = np.dot(sk, yk)  # 向量的点积
            yHy = np.dot(np.dot(yk, Hk), yk)  # yHy是标量
            Hk = Hk - 1.0 * Hy.reshape((n, 1)) * Hy / yHy + 1.0 * sk.reshape((n, 1)) * sk / sy

        k += 1
        x0 = x
    return x0, fun(x0), k,passing_dot


x0, fun0, k,x_arr = dfp(fun, gfun, hess, np.array([0., 0.]))
print (x0, fun0, k)
contour(X, Y, Z, x_arr)
plt.show()
