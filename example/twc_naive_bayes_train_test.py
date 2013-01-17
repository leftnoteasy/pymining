"""
this is a sample shows twc-naive-bayes train and test
"""

import math
import pickle
import sys,os
sys.path.append(os.path.join(os.getcwd(), '../'))

from pymining.math.matrix import Matrix
from pymining.math.text2matrix import Text2Matrix
from pymining.nlp.segmenter import Segmenter
from pymining.common.global_info import GlobalInfo
from pymining.common.configuration import Configuration 
from pymining.preprocessor.chisquare_filter import ChiSquareFilter
from pymining.classifier.twc_naive_bayes import TwcNaiveBayes

if __name__ == "__main__":
    config = Configuration.FromFile("conf/test.xml")
    GlobalInfo.Init(config, "__global__")
    txt2mat = Text2Matrix(config, "__matrix__")
    [trainx, trainy] = txt2mat.CreateTrainMatrix("data/train.txt")
    chiFilter = ChiSquareFilter(config, "__filter__")
    chiFilter.TrainFilter(trainx, trainy)
    [trainx, trainy] = chiFilter.MatrixFilter(trainx, trainy)
    nbModel = TwcNaiveBayes(config, "twc_naive_bayes")
    nbModel.Train(trainx, trainy)

    [testx, testy] = txt2mat.CreatePredictMatrix("data/test.txt")
    [testx, testy] = chiFilter.MatrixFilter(testx, testy)
    retY = nbModel.TestMatrix(testx, testy)
