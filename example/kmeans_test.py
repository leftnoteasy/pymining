"""
this is a example shows clustering algorithm - kmeans
data/cluster.200 is a 2-class data, 100 doc in each class
"""

import sys, os
sys.path.append(os.path.join(os.getcwd(), '../'))

from pymining.math.matrix import Matrix
from pymining.math.text2matrix import Text2Matrix
from pymining.nlp.segmenter import Segmenter
from pymining.clustering.kmeans import Kmeans
from pymining.common.global_info import GlobalInfo
from pymining.common.configuration import Configuration 

if __name__ == "__main__":
    config = Configuration.FromFile("conf/test.xml")
    GlobalInfo.Init(config, "__global__")
    txt2mat = Text2Matrix(config, "__matrix__")

    [x, y] = txt2mat.CreateTrainMatrix("data/cluster.200")

    kmeans = Kmeans()
    clusteringOut = kmeans.Cluster(x, 2)
    print clusteringOut
