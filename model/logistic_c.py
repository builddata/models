"""
LOGISTIC回归:分类模型
Logistic回归模型就是将线性回归的结果输入一个Sigmoid函数，将回归值映射到0~1，表示输出为类别的概率 Sigmoid：1/（1+E-z）
这里有一个前提假设，就是样本服从0-1分布，也就是伯努利分布n=1的情况（伯努利分布指的是对于随机变量X有, 参数为p(0<p<1)，
如果它分别以概率p和1-p取1和0为值。EX= p,DX=p(1-p)），用最大似然估计和梯度上升法学习Logistic Regression的模型参数\theta
sigmoid函数的输入记为z，即z=w0x0+w1x1+w2x2+...+wnxn，如果用向量表示即为z=wTx，它表示将这两个数值向量对应元素相乘然后累加起来。
其中，向量x是分类器的输入数据，w即为我们要拟合的最佳参数，从而使分类器预测更加准确。也就是说，logistic回归最重要的是要找到最佳的拟合参数
要最大化log-likelihood求参数\theta. 换一种角度理解，就是此时cost function J = - l(\theta)，我们需要最小化cost function 即- l(\theta)。
可以采用梯度上升法最大化log-likelihood
主要在流行病学中应用较多，比较常用的情形是探索某疾病的危险因素，根据危险因素预测某疾病发生的概率
缺点：Logistic回归对正负样本的分布比较敏感，所以要注意样本的平衡性
"""
from numpy import *


def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('logistic_c_testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


#评判一个优化算法的优劣的可靠方法是看其是否收敛，也就是说参数的值是否达到稳定值。此外，当参数值接近稳定时，仍然可能会出现一些小的周期性的波动。
# 这种情况发生的原因是样本集中存在一些不能正确分类的样本点(数据集并非线性可分)，
# 所以这些点在每次迭代时会引发系数的剧烈改变，造成周期性的波动。显然我们希望算法能够避免来回波动，从而收敛到某个值，并且收敛速度也要足够快
#随机梯度上升
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    #将数据集列表转为Numpy数组
    dataMat=array(dataMatrix)
    #获取数据集的行数和列数
    m,n=shape(dataMat)
    #初始化权值参数向量每个维度均为1
    weights=ones(n)
    #循环每次迭代次数
    for j in range(numIter):
        #获取数据集行下标列表
        dataIndex=range(m)
        #遍历行列表
        for i in range(m):
            #每次更新参数时设置动态的步长，且为保证多次迭代后对新数据仍然具有一定影响
            #添加了固定步长0.1
            alpha=4/(1.0+j+i)+0.1
            #随机获取样本：随机梯度下降通过每个样本来迭代更新一次参数，可能未遍历整个样本就已经找到最优解，大大提高了算法的收敛速度
            randomIndex=int(random.nuiform(0,len(dataIndex)))
            #计算当前sigmoid函数值
            h=sigmoid(dataMat[randomIndex]*weights)
            #计算权值更新
            #***********************************************
            error=classLabels-h
            weights=weights+alpha*error*dataMat[randomIndex]
            #***********************************************
            #选取该样本后，将该样本下标删除，确保每次迭代时只使用一次
            del(dataIndex[randomIndex])
    return weights
#--------------------------------------------------------------------------------------------------
#梯度提升算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxIteration = 1000
    weights = ones((n, 1))
    for k in range(maxIteration):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

#----------------------------------------------------------------------------------------------
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):  # get x,y locate at xcord ycord
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()


if __name__ == "__main__":
    dataArr, labelMat = loadDataSet()
    print(dataArr)
    print(labelMat)
    weights = gradAscent(dataArr, labelMat)
    print(weights)
    plotBestFit(weights.getA())
