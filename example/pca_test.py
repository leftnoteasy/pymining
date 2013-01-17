import sys, os
sys.path.append(os.path.join(os.getcwd(), '../'))

#import scipy comp
from scipy import *

#import pymining module
from pymining.math.pca import Pca
from pymining.math.matrix import Matrix
from pymining.math.text2matrix import Text2Matrix
from pymining.common.global_info import GlobalInfo
from pymining.common.configuration import Configuration

if __name__ == "__main__":
    config = Configuration.FromFile("conf/test.xml")
    GlobalInfo.Init(config, "__global__", False)
    txt2mat = Text2Matrix(config, "__matrix__", False)
    [trainx, trainy] = txt2mat.CreateTrainMatrix("data/train.txt")

    pca = Pca()
    pca.TrainPrinComp(trainx, 2, True, False)
    prinCompX = pca.GetPrinComp(trainx, "col")

    print prinCompX
