import math
import random
import pickle
import sys

from ..math.matrix import Matrix
from ..math.text2matrix import Text2Matrix
from ..nlp.segmenter import Segmenter
from ..common.global_info import GlobalInfo
from ..common.configuration import Configuration 

class Kmeans:
    """
    kmeans is a single-pass work, so don't have store model
    """
    def __init__(self):
        #self.curNode = config.GetChild(nodeName)
        self.__means = None
        self.__output = None
        self.__meansNorm2 = None

    """
    input x, and output a vector of every document belongs
    program runs pesudo-code:
    random-means-vector

    while (not converage):
        calculate for means
        check converage
    output result

    @k for k-output-cluster
    @return a vector y, len(y) = len(x)
    """
    def Cluster(self, x, k):
        """
        first, using twc-naive_bayes's tech
        assign x's value
        """
        for r in range(len(x.rows) - 1):
            sampleSum = 0.0
            for c in range(x.rows[r], x.rows[r + 1]):
                termId = x.cols[c]
                x.vals[c] = math.log(x.vals[c] + 1)
                x.vals[c] = x.vals[c] * GlobalInfo.idToIdf[termId]
                sampleSum += x.vals[c] * x.vals[c]

            #normalize it
            sampleSum = math.sqrt(sampleSum)
            for c in range(x.rows[r], x.rows[r + 1]):
                x.vals[c] = float(x.vals[c]) / sampleSum

        """
        second, runs kmeans clustering
        """
        #random-means-vector
        self.__InitMeans(x, k)

        #iterate
        converged = False
        while (not converged):
            converged = self.__CalculateMeans(x, k)

        #output
        return self.__output

    def __InitMeans(self, x, k):
        self.__means = [[0 for i in range(x.nCol)] for j in range(k)]

        self.__output = [0 for i in range(x.nRow)]
        self.__meansNorm2 = [0 for i in range(k)]
        for i in xrange(k):
            docId = random.randint(0, x.nRow - 1)
            for c in range(x.rows[docId], x.rows[docId + 1]):
                self.__means[i][x.cols[c]] = x.vals[c]
                self.__meansNorm2[i] += self.__means[i][x.cols[c]]**2
    
    def __CalculateMeans(self, x, k):
        meansSum = [[0 for i in range(x.nCol)] for j in range(k)]
        meansCount = [0 for i in range(k)]
        changed = 0

        #assign samples to means
        for r in range(len(x.rows) - 1):
            belongs = -1
            minCost = float(sys.maxint)
            
            #debug
            #print "new doc"
            for kk in range(k):
                cost = self.__CalculateCost(kk, x, r)

                if (cost < minCost):
                    minCost = cost
                    belongs = kk
            if self.__output[r] <> belongs:
                changed += 1
                self.__output[r] = belongs
            for c in range(x.rows[r], x.rows[r + 1]):
                meansSum[belongs][x.cols[c]] += x.vals[c]
            meansCount[belongs] += 1

        print "meansCount:", meansCount
        
        #calculate new means point
        for i in xrange(k):
            self.__meansNorm2[i] = 0
            for j in xrange(x.nCol):
                self.__means[i][j] = meansSum[i][j] / meansCount[i]
                self.__meansNorm2[i] += self.__means[i][j]**2

        if float(changed) / x.nRow <= 0.01:
            return True
        else:
            return False

    """
    using Euclidean distance and a simple trick:
    when calculate dist of dense vector(means vector) and
         sparse vector(sample vector):
    dist(dV, sV) = sqrt(dV[0]^2 + dV[1]^2..dV[n]^2) +
        (dV[k0] - sV[0])^2 + (dV[k1] - sV[1])^2 ... -
        (dV[k0]^2 + dV[k1]^2 ... dV[km]^2))
    """
    def __CalculateCost(self, kk, x, r):
        cost = self.__meansNorm2[kk]
        #print "meansNorm:", cost
        for c in range(x.rows[r], x.rows[r + 1]):
            termId = x.cols[c]

            #debug
            #print x.vals[c], " ", self.__means[kk][termId]
            cost += (x.vals[c] - self.__means[kk][termId]) * (x.vals[c] - self.__means[kk][termId]) - self.__means[kk][termId] * self.__means[kk][termId]

        #print "cost:",cost

        return math.sqrt(cost + 1e-8)
