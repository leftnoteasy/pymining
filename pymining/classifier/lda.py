import math
import pickle

import numpy
import scipy
import scipy.linalg

from ..math.matrix import Matrix
from ..math.text2matrix import Text2Matrix
from ..math.scipy_interface import ScipyInterface
from ..nlp.segmenter import Segmenter
from ..common.global_info import GlobalInfo
from ..common.configuration import Configuration 

class Lda:
    def __init__(self, config, nodeName, loadFromFile = False):
        self.sourceDim = 0
        self.w = None
        if (loadFromFile):
            #add code
            pass

    def Train(self, x, y, k):
        #make yy unique
        yy = set(y)
        yy = list(yy)
        yy.sort()
        if (len(yy) <> 2):
            print "ERROR:input matrix should be 2-class classification problem"
            raise

        #check nDim
        if (k <= 0) or (k > x.nCol):
            print "k MUST > 0 and < x.nCol"
            raise
        self.sourceDim = x.nCol

        #calculate m1, m2
        denseMat = ScipyInterface.CsrToDense(x)
        print "denseMat_shape:", denseMat.shape
        m1 = numpy.zeros(denseMat.shape[1])
        m2 = numpy.zeros(denseMat.shape[1])
        for i in range(denseMat.shape[0]):
            if (y[i] == yy[0]):
                m1 += denseMat[i].ravel()
            else:
                m2 += denseMat[i].ravel()

        #calculate s1, s2, sw
        s1 = numpy.zeros((denseMat.shape[1], denseMat.shape[1]))
        s2 = numpy.zeros((denseMat.shape[1], denseMat.shape[1]))
        for i in range(denseMat.shape[0]):
            if (y[i] == yy[0]):
                vec = denseMat[i].ravel() - m1
                s1 += numpy.dot(vec, vec.transpose())
            else:
                vec = denseMat[i].ravel() - m2
                s2 += numpy.dot(vec, vec.transpose())
        sw = s1 + s2
        
        #calculate w
        sw = scipy.linalg.pinv(sw)
        self.w = numpy.dot(sw, m1 - m2)
        self.w = self.w[0:k]
        self.w = self.w / scipy.linalg.norm(self.w)

    def Test(self, x, y):
        #add code
        pass

