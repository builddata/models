"""
决策树CART_:回归
CART用GINI指数来决定如何分裂,基尼系数代表了模型的不纯度，基尼系数越小，不纯度越低，特征越好
CART分类树算法每次仅对某个特征的值进行二分，而不是多分，这样CART分类树算法建立起来的是二叉树，而不是多叉树
CART分类树算法对离散值的处理，采用的思路：CART是不停的二分。
会考虑把特征A分成{A1}和{A2,A3}、{A2}和{A1,A3}、{A3}和{A1,A2}三种情况，找到基尼系数最小的组合

分类树与回归树的区别在样本的输出，如果样本输出是离散值，这是分类树；样本输出是连续值，这是回归树
分类模型：采用基尼系数的大小度量特征各个划分点的优劣。
回归模型: 采用平方误差最小化准则进行特征选择，把最终各个叶子中所有样本对应结果值的均值或者中位数当作预测的结果输出
CART分类树采用叶子节点里概率最大的类别作为当前节点的预测类别。回归树输出不是类别，采用叶子节点的均值或者中位数来预测输出结果

优点：

简单直观，生成的决策树很直观。
基本不需要预处理，不需要提前归一化和处理缺失值。
使用决策树预测的代价是O(log2m)。m为样本数。
既可以处理离散值也可以处理连续值。很多算法只是专注于离散值或者连续值。
可以处理多维度输出的分类问题。
相比于神经网络之类的黑盒分类模型，决策树在逻辑上可以很好解释。
可以交叉验证的剪枝来选择模型，从而提高泛化能力。
对于异常点的容错能力好，健壮性高。

缺点：

决策树算法非常容易过拟合，导致泛化能力不强。可以通过设置节点最少样本数量和限制决策树深度来改进。
决策树会因为样本发生一点的改动，导致树结构的剧烈改变。这个可以通过集成学习之类的方法解决。
寻找最优的决策树是一个NP难题，我们一般是通过启发式方法，容易陷入局部最优。可以通过集成学习的方法来改善。
有些比较复杂的关系，决策树很难学习，比如异或。这个就没有办法了，一般这种关系可以换神经网络分类方法来解决。
如果某些特征的样本比例过大，生成决策树容易偏向于这些特征。这个可以通过调节样本权重来改善。
"""
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import treePlotter
def loadDataSet(filename):
    '''
    输入：文件的全路径
    功能：将输入数据保存在datamat
    输出：datamat
    '''
    fr = open(filename)
    datamat = []
    for line in fr.readlines():
        cutLine = line.strip().split('\t')
        floatLine = map(float,cutLine)
        datamat.append(floatLine)
    return datamat

def binarySplitDataSet(dataset,feature,value):
    '''
    输入：数据集，数据集中某一特征列，该特征列中的某个取值
    功能：将数据集按特征列的某一取值换分为左右两个子数据集
    输出：左右子数据集
    '''
    matLeft = dataset[np.nonzero(dataset[:,feature] <= value)[0],:]
    matRight = dataset[np.nonzero(dataset[:,feature] > value)[0],:]
    return matLeft,matRight

#--------------回归树所需子函数---------------#

def regressLeaf(dataset):
    '''
    输入：数据集
    功能：求数据集输出列的均值
    输出：对应数据集的叶节点
    '''
    return np.mean(dataset[:,-1])


def regressErr(dataset):
    '''
    输入：数据集(numpy.mat类型)
    功能：求数据集划分左右子数据集的误差平方和之和
    输出: 数据集划分后的误差平方和
    '''
    #由于回归树中用输出的均值作为叶节点，所以在这里求误差平方和实质上就是方差
    return np.var(dataset[:,-1]) * np.shape(dataset)[0]
def regressData(filename):
    fr = open(filename)
    return pickle.load(fr)
#--------------回归树子函数  END  --------------#

def chooseBestSplit(dataset,leafType=regressLeaf,errType=regressErr,threshold=(1,4)):#函数做为参数，挺有意思
    thresholdErr = threshold[0];thresholdSamples = threshold[1]
    #当数据中输出值都相等时，feature = None,value = 输出值的均值（叶节点）
    if len(set(dataset[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataset)
    m,n = np.shape(dataset)
    Err = errType(dataset)
    bestErr = np.inf; bestFeatureIndex = 0; bestFeatureValue = 0
    for featureindex in range(n-1):
        for featurevalue in dataset[:,featureindex]:
            matLeft,matRight = binarySplitDataSet(dataset,featureindex,featurevalue)
            if (np.shape(matLeft)[0] < thresholdSamples) or (np.shape(matRight)[0] < thresholdSamples):
                continue
            temErr = errType(matLeft) + errType(matRight)
            if temErr < bestErr:
                bestErr = temErr
                bestFeatureIndex = featureindex
                bestFeatureValue = featurevalue
    #检验在所选出的最优划分特征及其取值下，误差平方和与未划分时的差是否小于阈值，若是，则不适合划分
    if (Err - bestErr) < thresholdErr:
        return None,leafType(dataset)
    matLeft,matRight = binarySplitDataSet(dataset,bestFeatureIndex,bestFeatureValue)
    #检验在所选出的最优划分特征及其取值下，划分的左右数据集的样本数是否小于阈值，若是，则不适合划分
    if (np.shape(matLeft)[0] < thresholdSamples) or (np.shape(matRight)[0] < thresholdSamples):
        return None,leafType(dataset)
    return bestFeatureIndex,bestFeatureValue


def createCARTtree(dataset,leafType=regressLeaf,errType=regressErr,threshold=(1,4)):

    '''
    输入：数据集dataset，叶子节点形式leafType：regressLeaf（回归树）、modelLeaf（模型树）
         损失函数errType:误差平方和也分为regressLeaf和modelLeaf、用户自定义阈值参数：
         误差减少的阈值和子样本集应包含的最少样本个数
    功能：建立回归树或模型树
    输出：以字典嵌套数据形式返回子回归树或子模型树或叶结点
    '''
    feature,value = chooseBestSplit(dataset,leafType,errType,threshold)
    #当不满足阈值或某一子数据集下输出全相等时，返回叶节点
    if feature == None: return value
    returnTree = {}
    returnTree['bestSplitFeature'] = feature
    returnTree['bestSplitFeatValue'] = value
    leftSet,rightSet = binarySplitDataSet(dataset,feature,value)
    returnTree['left'] = createCARTtree(leftSet,leafType,errType,threshold)
    returnTree['right'] = createCARTtree(rightSet,leafType,errType,threshold)
    return returnTree

#----------回归树剪枝函数----------#
def isTree(obj):#主要是为了判断当前节点是否是叶节点
    return (type(obj).__name__ == 'dict')

def getMean(tree):#树就是嵌套字典
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right'])/2.0

def prune(tree, testData):
    if np.shape(testData)[0] == 0: return getMean(tree)#存在测试集中没有训练集中数据的情况
    if isTree(tree['left']) or isTree(tree['right']):
        leftTestData, rightTestData = binarySplitDataSet(testData,tree['bestSplitFeature'],tree['bestSplitFeatValue'])
    #递归调用prune函数对左右子树,注意与左右子树对应的左右子测试数据集
    if isTree(tree['left']): tree['left'] = prune(tree['left'],leftTestData)
    if isTree(tree['right']): tree['right'] = prune(tree['right'],rightTestData)
    #当递归搜索到左右子树均为叶节点时，计算测试数据集的误差平方和
    if not isTree(tree['left']) and not isTree(tree['right']):
        leftTestData, rightTestData = binarySplitDataSet(testData,tree['bestSplitFeature'],tree['bestSplitFeatValue'])
        errorNOmerge = sum(np.power(leftTestData[:,-1] - tree['left'],2)) +sum(np.power(rightTestData[:,-1] - tree['right'],2))
        errorMerge = sum(np.power(testData[:,1] - getMean(tree),2))
        if errorMerge < errorNOmerge:
            print ('Merging')
            return getMean(tree)
        else: return tree
    else: return tree

#---------回归树剪枝END-----------#

#-----------模型树子函数-----------#
def linearSolve(dataset):
    m,n = np.shape(dataset)
    X = np.mat(np.ones((m,n)));Y = np.mat(np.ones((m,1)))
    X[:,1:n] = dataset[:,0:(n-1)]
    Y = dataset[:,-1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of threshold')
        ws = xTx.I * (X.T * Y)
        return ws, X,Y

def modelLeaf(dataset):
    ws,X,Y = linearSolve(dataset)
    return ws

def modelErr(dataset):
    ws,X,Y = linearSolve(dataset)
    yHat = X * ws
    return sum(np.power(Y - yHat,2))

#------------模型树子函数END-------#

#------------CART预测子函数------------#

def regressEvaluation(tree, inputData):
    #只有当tree为叶节点时，才会输出
    return float(tree)

def modelTreeEvaluation(model,inputData):
    #inoutData为采样数为1的特征行向量
    n = np.shape(inputData)
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1] = inputData
    return float(X * model)

def treeForeCast(tree, inputData, modelEval = regressEvaluation):
    if not isTree(tree): return modelEval(tree,inputData)
    if inputData[tree['bestSplitFeature']] <= tree['bestSplitFeatValue']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inputData,modelEval)
        else:
            return modelEval(tree['left'],inputData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inputData,modelEval)
        else:
            return modelEval(tree['right'],inputData)

def createForeCast(tree,testData,modelEval=regressEvaluation):
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat = treeForeCast(tree,testData[i],modelEval)
    return yHat

#-----------CART预测子函数 END------------#

if __name__ == '__main__':

    trainfilename = 'e:\\python\\ml\\trainDataset.txt'
    testfilename = 'e:\\python\\ml\\testDataset.txt'

    trainDataset = regressData(trainfilename)
    testDataset = regressData(testfilename)

    cartTree = createCARTtree(trainDataset,threshold=(1,4))
    pruneTree=prune(cartTree,testDataset)
    treePlotter.createPlot(cartTree)
    y=createForeCast(cartTree,np.mat([0.3]),modelEval=regressEvaluation)
