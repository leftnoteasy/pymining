import scipy
import numpy
import scipy.sparse.linalg

#import pymining module
from matrix import *
from scipy_interface import *

class Pca:
    def __init__(self):
        self.isTrained = False
        self.haveColSpace = False
        self.haveRowSpace = False
        self.colSpaceTrans = None
        self.rowSpaceTrans = None

    """
    this function get principal component of a given matrix
    @x is the input matrix
    @k is get top-k dimension of pc
    @rowSpace = True, is get U*S
    @colSpace = True, is get S*V'
    @return:
        if rowSpace = True and colSpace = True, return (U*S, (S*V')')
        if rowSpace = True and colSpace = False, return U*S
        if rowSpace = False and colSpace = True, return (S*V')'
    """
    def TrainPrinComp(self, x, k, colSpace = True, rowSpace = False):
        if (not colSpace) and (not rowSpace):
            print "colSpace and rowSpace both == False, what you want to do?"
            raise

        self.haveColSpace = colSpace
        self.haveRowSpace = rowSpace

        #do incompelate svd
        csrMat = ScipyInterface.MatrixToCsr(x)

        #get u,s,v'
        (u,s,vt) = scipy.sparse.linalg.svds(csrMat, k)
        s = numpy.diag(s)

        if (colSpace):
            self.colSpaceTrans = numpy.dot(s, vt).transpose()
        elif (rowSpace):
            self.rowSpaceTrans = numpy.dot(u,s)

        self.isTrained = True

    """
    using trained col/row space tranform to do transform
    @spaceParam in {"row", "col"}
    @return
    if spaceParam == "row", return U*S * sourceMat
    if spaceParam == "col", return sourceMat * (S*V')'
    """
    def GetPrinComp(self, sourceMat, spaceParam):
        if not self.isTrained:
            print "using pca before train it!"
            raise

        if (spaceParam <> "row") and (spaceParam <> "col"):
            print "spaceParam shoud in {\"row\", \"col\"}"
            raise
        if (spaceParam == "row"):
            return numpy.dot(self.rowSpaceTrans, \
                ScipyInterface.MatrixToCsr(sourceMat).todense())
        else:
            return numpy.dot(ScipyInterface.MatrixToCsr(sourceMat).todense(), \
                self.colSpaceTrans)
