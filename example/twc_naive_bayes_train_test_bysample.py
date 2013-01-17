#encoding=utf-8

"""
this is a sample shows twc-naive-bayes train and test by a sample in str
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
import sys, os
sys.path.append(os.path.join(os.getcwd(), '../'))

import math
import pickle
import sys

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
    nbModel = TwcNaiveBayes(config, "twc_naive_bayes")
    nbModel.Train(trainx, trainy)

    inputStr = "仅售59元！原价108元的花园巴西烤肉自助餐一人次任吃（蛇口店、购物公园店全时段通用），另赠送两张10元现金抵用券！邀请好友返利10元！"
    [cols, vals] = txt2mat.CreatePredictSample(inputStr.decode("utf-8"))
    retY = nbModel.TestSample(cols, vals)
    print retY
