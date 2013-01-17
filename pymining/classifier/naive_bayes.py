import math
import pickle

from ..math.matrix import Matrix
from ..math.text2matrix import Text2Matrix
from ..nlp.segmenter import Segmenter
from ..common.global_info import GlobalInfo
from ..common.configuration import Configuration 

class NaiveBayes:
    def __init__(self, config, nodeName, loadFromFile = False):
        #store variable(term)'s likelihood to each class
        self.vTable = []
        #store prior of each class
        self.cPrior = []
        #store isTrained by data
        self.trained = loadFromFile

        self.curNode = config.GetChild(nodeName)
        self.modelPath = self.curNode.GetChild("model_path").GetValue()
        self.logPath = self.curNode.GetChild("log_path").GetValue()

        if (loadFromFile):
            f = open(self.modelPath, "r")
            modelStr = pickle.load(f)
            [self.vTable, self.cPrior] = pickle.loads(modelStr)
            f.close()

    def Train(self, x, y):
        #check parameters
        if (x.nRow <> len(y)):
            print "ERROR!, x.nRow should == len(y)"
            return False

        #calculate prior of each class
        #1. init cPrior:
        yy = set(y)
        yy = list(yy)
        yy.sort()
        self.cPrior = [0 for i in range(yy[len(yy) - 1] + 1)]

        #2. fill cPrior
        for i in y:
            self.cPrior[i] += 1

        #calculate likehood of each term
        #1. init vTable:
        self.vTable = [[0 for i in range(len(self.cPrior))] \
                for j in range(x.nCol)]

        #2. fill vTable
        for r in range(x.nRow):
            for i in range(x.rows[r], x.rows[r + 1]):
                self.vTable[x.cols[i]][y[r]] += 1
        
        #normalize vTable
        for i in range(x.nCol):
            for j in range(len(self.cPrior)):
                if (self.cPrior[j] > 1e-10):
                    self.vTable[i][j] /= float(self.cPrior[j])
        
        #normalize cPrior
        for i in range(len(self.cPrior)):
            self.cPrior[i] /= float(len(y))

        self.trained = True

        #dump model path
        f = open(self.modelPath, "w")
        modelStr = pickle.dumps([self.vTable, self.cPrior], 1)
        pickle.dump(modelStr, f)
        f.close()

        return True

    def TestSample(self, cols, vals):
        #check parameter
        if (not self.trained):
            print "Error!, not trained!"
            return False
        if (len(cols) <> len(vals)):
            print "Error! len of cols should == len of vals"
            return False

        #calculate best p
        targetP = []
        maxP = -1000000000
        for target in range(len(self.cPrior)):
            curP = 0
            curP += math.log(self.cPrior[target])
            
            for c in range(0, len(cols)):
                if (self.vTable[cols[c]][target] == 0):
                    curP += math.log(1e-7)
                else:
                    curP += math.log(self.vTable[cols[c]][target])
                #debug
                #if (self.logPath <> ""):
                #    term = GlobalInfo.idToTerm[cols[c]]
                #    prob = math.log(self.vTable[cols[c]][target] + 1e-7) 
                #    f.write(term.encode("utf-8") + ":" + str(cols[c]) + ":" + str(prob) + "\n")
            
            targetP.append(curP)
            if (curP > maxP):
                bestY = target
                maxP = curP

        #normalize probable
        ret = []
        total = 0
        for i in range(len(targetP)):
            total += math.exp(targetP[i])
        for i in range(len(targetP)):
            ret.append((i, math.exp(targetP[i]) / total))

        return tuple(ret)

    def Test(self, x, y):
        #check parameter
        if (not self.trained):
            print "Error!, not trained!"
            return False
        
        if (x.nRow != len(y)):
            print "Error! x.nRow should == len(y)"
            return False

        retY = []
        correct = 0

        if (self.logPath <> ""):
            f = open(self.logPath, "w")

        #predict all doc one by one
        for r in range(x.nRow):
            bestY = -1
            maxP = -1000000000

            #debug
            if (self.logPath <> ""):
                f.write("\n ===============new doc=================")

            #calculate best p
            for target in range(len(self.cPrior)):
                curP = 0
                if (self.cPrior[target] > 1e-8):
                    curP += math.log(self.cPrior[target])
                else:
                    curP += math.log(1e-8)
                
                #debug
                #if (self.logPath <> ""):
                #    f.write("<target> : " + str(target) + "\n")

                for c in range(x.rows[r], x.rows[r + 1]):
                    if (self.vTable[x.cols[c]][target] == 0):
                        curP += math.log(1e-7)
                    else:
                        curP += math.log(self.vTable[x.cols[c]][target])

                    #debug
                    #if (self.logPath <> ""):
                    #    term = GlobalInfo.idToTerm[x.cols[c]]
                    #    prob = math.log(self.vTable[x.cols[c]][target] + 1e-7) 
                    #    f.write(term.encode("utf-8") + ":" + str(x.cols[c]) + ":" + str(prob) + "\n")

                if (curP > maxP):
                    bestY = target
                    maxP = curP

                #debug
                #if (self.logPath <> ""):
                #    f.write("curP:" + str(curP) + "\n")

            if (bestY < 0):
                print "best y < 0, error!"
                return False
            if (bestY == y[r]):
                correct += 1
            #debug
            else:
                if (self.logPath <> ""):
                    f.write("predict error!")
            retY.append(bestY)

        if (self.logPath <> ""):
            f.close()
        
        return [retY, float(correct) / len(retY)]
