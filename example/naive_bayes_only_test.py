"""
this is a example shows load saved model and test new samples
notice in GlobalInfo.Init and other initialize functions
the second parameter - LoadFromFile is true, means load trained result
"""
import sys, os
sys.path.append(os.path.join(os.getcwd(), '../'))

from pymining.math.matrix import Matrix
from pymining.math.text2matrix import Text2Matrix
from pymining.nlp.segmenter import Segmenter
from pymining.common.global_info import GlobalInfo
from pymining.common.configuration import Configuration
from pymining.preprocessor.chisquare_filter import ChiSquareFilter
from pymining.classifier.naive_bayes import NaiveBayes

if __name__ == "__main__":
    config = Configuration.FromFile("conf/test.xml")
    GlobalInfo.Init(config, "__global__", True)
    txt2mat = Text2Matrix(config, "__matrix__", True)
    chiFilter = ChiSquareFilter(config, "__filter__", True)

    nbModel = NaiveBayes(config, "naive_bayes", True)

    [testx, testy] = txt2mat.CreatePredictMatrix("data/test.txt")
    [testx, testy] = chiFilter.MatrixFilter(testx, testy)
    [resultY, precision] = nbModel.Test(testx, testy)
    
    print precision
