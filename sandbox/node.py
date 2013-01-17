import matrix
import math

from matrix import *
from math import *

class Node:
    def __init__(self, x, y, baggingDict):
        self.x = x
        self.y = y
        self.featureSet = set()
        self.isLeaf = False
        self.target = -1
        self.variable = -1
        self.leftChild = None
        self.rightChild = None
        self.baggingDict = baggingDict
        for i in range(0, len(self.x.cols)):
            self.featureSet.add(self.x.cols[i])

    def Learn(self):
        #check is split enough(all cate in current node is same)
        if (len(self.y) == 0):
            self.isLeaf = True
            self.target = -1
            return 
        tmp = self.y[0]
        isSame = True
        for i in range(1, len(self.y)):
            if tmp != self.y[i]:
                isSame = False
                break

        #if is same, set current node to leaf node
        if isSame:
            self.isLeaf = True
            self.target = tmp
            return


        #find max-cate num
        maxCate = -1
        for i in range(0, len(self.y)):
            if (self.y[i] > maxCate):
                maxCate = self.y[i]
        
        #else, calculate information-gain(IG(Y|X))
        bestGain = 11000000000
        bestSplit = -1
        majorTarget = -1
        for feat in self.featureSet:  
            #get how much cate in current y
            yZeros = [] #when x = 0, counts of y's value
            yOnes = []  #when x = 1, counts of y's value
            xZero = 0   #number of sample[feat] == 0
            xOne = 0    #number of sample[feat] == 1
            for j in range(0, maxCate + 1):
                yZeros.append(0)
                yOnes.append(0)

            #IG(Y|X) = H(Y) - H(Y|X), x is current feature
            #H(Y) = -sigma pj*log(pj,2)
            #          j
            #H(X) = H(Y|x = 0)P(x = 0) + H(Y|x = 1)P(x = 1)
            for sample in range(0, len(self.x.rows) - 1):
                #print "get:", sample, ",", feat
                value = self.x.Get(sample, feat)
                if (value == 0):
                   yZeros[self.y[sample]] += 1
                   xZero += 1
                else:
                   yOnes[self.y[sample]] += 1
                   xOne += 1

            #using yZeros and yOnes Get IG(Y|X)
            #calculate H(Y|x = 0)
            zeroGain = 0
            for j in range(0, maxCate + 1):
                if (yZeros[j] > 0):
                    p = yZeros[j] * 1.00 / xZero
                    zeroGain += -1 * p * log(p, 2)
            #print "zero gain:", zeroGain
            #calculate H(Y|x = 0) * p(x = 0)
            zeroGain *= xZero * 1.00 / (xZero + xOne)
            #calculate H(Y|x = 1)
            oneGain = 0
            for i in range(0, maxCate + 1):
                if (yOnes[i] > 0):
                    p = yOnes[i] * 1.00 / xOne
                    oneGain += -1 * p * log(p, 2)

            #calculate major target
            maxTar = -1
            for j in range(0, maxCate + 1):
                if (yZeros[j] + yOnes[j] > 0):
                    maxTar = j

            #calculate H(Y|x = 1) * p(x = 1)
            oneGain *= xOne * 1.00 / (xZero + xOne)
            #print "one gain:", oneGain
            if (zeroGain + oneGain < bestGain):
                bestGain = zeroGain + oneGain
                bestSplit = feat
                majorTarget = maxTar
            #debug
            #print "yOnes:", yOnes
            #print "yZeros:", yZeros
            #print "xOnes:", xOne
            #print "xZeros:", xZero

        #using bestSplit split x,y to left, right child
        leftRows = [0]
        rightRows = [0]
        leftCols = []
        rightCols = []
        leftVals = []
        rightVals = []
        leftY = []
        rightY = []
        self.variable = bestSplit

        #split training set to left and right child 
        for sample in range(0, self.x.nRow):
            if self.x.Get(sample, bestSplit):
                rightRows.append(rightRows[len(rightRows) - 1] + \
                self.x.rows[sample + 1] - self.x.rows[sample])
                for i in range(self.x.rows[sample], self.x.rows[sample + 1]):
                    rightCols.append(self.x.cols[i])
                    rightVals.append(self.x.vals[i])
                rightY.append(self.y[sample])
            else:
                leftRows.append(leftRows[len(leftRows) - 1] + \
                self.x.rows[sample + 1] - self.x.rows[sample])
                for i in range(self.x.rows[sample], self.x.rows[sample + 1]):
                    leftCols.append(self.x.cols[i])
                    leftVals.append(self.x.vals[i])
                leftY.append(self.y[sample])
        leftMat = Matrix(leftRows, leftCols, leftVals)
        rightMat = Matrix(rightRows, rightCols, rightVals)
        #print "leftChild.mat:",len(leftMat.rows), " ", len(leftY)
        #print "rightChild.mat:",len(rightMat.rows), " ", len(rightY)
        if (len(leftY) == 0) or (len(rightY) == 0):
            #print "all input is same"
            self.isLeaf = True
            self.target = majorTarget
            self.x = None
            self.y = []
            return

        self.leftChild = Node(leftMat, leftY, self.baggingDict)
        self.rightChild = Node(rightMat, rightY, self.baggingDict)
        self.leftChild.Learn()
        self.rightChild.Learn()

        #delete current data
        self.x = None
        self.y = []

    def Predict(self, sample):
        if (self.isLeaf):
            return self.target
        else:
            #print "self.variable",self.variable
            if (sample.Get(0, self.baggingDict[self.variable])):
                if (self.leftChild != None):
                    return self.leftChild.Predict(sample)
                else:
                    return -1
            else:
                if (self.rightChild != None):
                    return self.rightChild.Predict(sample)
                else:
                    return -1

