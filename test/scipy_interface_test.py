import sys, os
sys.path.append(os.path.join(os.getcwd(), '../'))

from pymining.math.matrix import *
from pymining.math.scipy_interface import *
from scipy.sparse import *
from scipy import *

def convert_test(rows, cols, vals):
    mat = Matrix(rows, cols, vals)
    csr = ScipyInterface.MatrixToCsr(mat)
    print csr.todense()

    mat = ScipyInterface.CsrToMatrix(csr)
    csr = ScipyInterface.MatrixToCsr(mat)
    print csr.todense()

if __name__ == "__main__":
    print "==================="
    rows = [0, 2, 3, 6]
    cols = [0, 1, 2, 0, 1, 2]
    vals = [1, 1, 1, 1, 1, 1]
    convert_test(rows, cols, vals)

    print "==================="
    rows = [0, 2, 2, 2, 3]
    cols = [0, 1, 2]
    vals = [1.1, 1.1, 1.1]
    convert_test(rows, cols, vals)

    print "==================="
    rows = [0, 0, 0, 2, 2, 2, 3]
    cols = [0, 1, 2]
    vals = [1.1, 1.1, 1.1]
    convert_test(rows, cols, vals)

