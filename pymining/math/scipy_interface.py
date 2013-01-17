from scipy.sparse import *
from scipy import *
import numpy
from matrix import Matrix

"""
methods for converting between pymining-Matrix and scipy.sparse.csr_matrix
"""
class ScipyInterface:
    """
    convert scipy.sparse.csr_matrix to pymining-Matrix
    @return pymining.math.Matrix
    """
    @staticmethod
    def CsrToMatrix(csrSource):
       (csrRows, csrCols) = csrSource.nonzero()
       cols = csrCols.tolist()
       rows = [0]
       vals = []
       for i in range(len(csrRows)):
           vals.append(csrSource[csrRows[i], csrCols[i]])
       idx = 0
       for i in range(csrSource.shape[0]):
           curCount = 0
           if (idx < len(csrRows)):
               while (csrRows[idx] == i):
                   curCount += 1
                   idx += 1
                   if (idx >= len(csrRows)):
                      break
           rows.append(rows[len(rows) - 1] + curCount)
       return Matrix(rows, cols, vals, csrSource.shape[0], csrSource.shape[1])

    """
    convert a sparse matrix to dense matrix in numpy
    """
    @staticmethod
    def CsrToDense(src):
        return ScipyInterface.MatrixToCsr(src).toarray()

    """
    convert pymining-Matrix to scipy.sparse.csr_matrix
    @return scipy.sparse.csr_matrix
    """
    @staticmethod
    def MatrixToCsr(mat):
        return csr_matrix((array(mat.vals, numpy.float64), array(mat.cols), array(mat.rows))\
            , shape = (mat.nRow, mat.nCol))
