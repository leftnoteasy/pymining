#encoding=utf8
"""
this is a example shows train model using a corpus, and test a sample in str
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
    GlobalInfo.Init(config, "__global__")
    txt2mat = Text2Matrix(config, "__matrix__")
    [trainx, trainy] = txt2mat.CreateTrainMatrix("data/train.txt")
    chiFilter = ChiSquareFilter(config, "__filter__")
    chiFilter.TrainFilter(trainx, trainy)

    nbModel = NaiveBayes(config, "naive_bayes")
    nbModel.Train(trainx, trainy)
    
    inputStr = "仅售28元！原价698元的康迩福韩国美容美体中心的韩国特色美容套餐1份（紫莱花园店、时代奥城店2店通用）：韩国特色面部SPA护理1次+韩国特色面部瘦脸加毛孔净化1次+韩国特色水"
    [cols, vals] = txt2mat.CreatePredictSample(inputStr.decode("utf-8"))
    [cols, vals] = chiFilter.SampleFilter(cols, vals)
    probTuple = nbModel.TestSample(cols, vals)
    print probTuple
