import matrix
import node
import math

from math import *
from matrix import *
from node import *

class RandomForest:
    #build a forest consist of nTree trees
    #and each tree have (int)(ratio * nFeatures) variables
    def __init__(self, x, y, nTree, ratio):
        self.nTree = nTree
        self.nFeature = int(x.nCol * ratio)
        self.x = x
        self.y = y
        self.trees = []

    def Learn(self):
        #get transpose of x
        tX = self.x.Transpose(self.x.nCol)
        #print self.x.rows
        #print self.x.cols

        print "after transpose"

        #print tX.rows
        #print tX.cols

        #spawn forest
        for i in range(0, self.nTree):
            [subX, baggingDict] = Matrix.BaggingFromMatrix(tX, self.nFeature)

            #print "after bagging:"
            #print self.nFeature
            #print subX.rows
            #print subX.cols
            #print subX.vals

            node = Node(subX, self.y, baggingDict)
            node.Learn()
            self.trees.append(node)

        print "ntree in train:", len(self.trees)

    def Predict(self, sample):
        print "enter predict:"
        results = {}
        for i in range(0, self.nTree):
            result = self.trees[i].Predict(sample)
            if not results.has_key(result):
                results[result] = 0
            else:
                results[result] += 1

        bestResult = -1
        maxNum = -1
        for result in results:
            if (results[result] > maxNum) and (result != -1):
                maxNum = results[result]
                bestResult = result

        print results

        return bestResult
