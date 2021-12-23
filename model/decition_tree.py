from math import log
import operator

# 计算香农熵
class Tree:
    def __init__(self):
        super()
    def calcShannonEnt(self, dataSet):
        num = len(dataSet)
        labelCounts = {}
        for fVec in dataSet:
            currentLabel = fVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / num
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    #按照特征划分数据集，特征的位置为index
    def splitDataSet(self, dataSet, index, value):
        retDataSet = []
        for featVec in dataSet:
            if featVec[index] == value:
                reducedFeatVec = featVec[:index]
                reducedFeatVec.extend(featVec[index+1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    #寻找信息增益最大的特征
    def chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calcShannonEnt(dataSet)
        bestInfoGain, bestFeature = 0.0, -1
        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0.0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy +=prob * self.calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            if (infoGain >= bestInfoGain):#这里注意，取等号，只有1个特征为时，可能无信息增加。
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    # 如果分类不唯一，采用多数表决方法，决定叶子的分类
    def majorityCnt(self, classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        SortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
        return SortedClassCount[0][0]

    # 创建决策树代码
    def createTree(self, dataSet, labels):
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0] == len(classList)):#类别完全相同，无需划分，一类
            return classList[0]
        if len(dataSet[0]) == 1: #处理了所有特征，依旧没有完全划分，返回多数表决结果
            return self.majorityCnt(classList)
        bestFeat = self.chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        del labels[bestFeat]
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:#利用递归构建决策树
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), subLabels)
        return myTree

    def createDataSet(self):
        dataSet = [
            [1,1,"yes"],
            [1,0,"no"],
            [0,1,"no"],
            [0,1,"no"]
        ]
        labels =["no surfacing", "flippers"]
        return dataSet, labels
    def decisiontreeclassify(self, inputTree, featLabels, testVec):
        firstStr = list(inputTree.keys())[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self.decisiontreeclassify(secondDict[key],featLabels,testVec)
                else:
                    classLabel = secondDict[key]
        return classLabel
if __name__ == "__main__":
    tree = Tree()
    myDat, myLabels =tree.createDataSet()
    inputTree = tree.createTree(myDat, myLabels)
    featLabels = ['no surfacing','flippers']
    print(inputTree)
    print(tree.decisiontreeclassify( inputTree, featLabels, [1,0]))
    print(tree.decisiontreeclassify( inputTree, featLabels, [1,1]))
