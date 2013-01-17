import pickle

from ..math.matrix import Matrix
from ..math.text2matrix import Text2Matrix
from ..nlp.segmenter import Segmenter
from ..common.global_info import GlobalInfo
from ..common.configuration import Configuration

class ChiSquareFilter:
    def __init__(self, config, nodeName, loadFromFile = False):
        self.curNode = config.GetChild(nodeName)
        self.rate = float(self.curNode.GetChild("rate").GetValue())
        self.method = self.curNode.GetChild("method").GetValue()
        self.logPath = self.curNode.GetChild("log_path").GetValue()
        self.modelPath = self.curNode.GetChild("model_path").GetValue()
        self.idMap = None
        self.trained = loadFromFile
        if (loadFromFile):
            f = open(self.modelPath, "r")
            modelStr = pickle.load(f)
            [self.idMap] = pickle.loads(modelStr)
            f.close()

    def SampleFilter(self, cols, vals):
        if (not self.trained):
            print "train filter before test"
            return False

        #check parameter
        if (len(cols) <> len(vals)):
            print "length of cols should equals length of vals"
            return False

        #filter sample
        newCols = []
        newVals = []
        for c in range(0, len(cols)):
            if self.idMap[cols[c]] >= 0:
                newCols.append(self.idMap[cols[c]])
                newVals.append(vals[c])

        return [cols, vals]

    """
    filter given x,y by blackList
    x's row should == y's row
    @return newx, newy filtered
    """
    def MatrixFilter(self, x, y):
        if (not self.trained):
            print "train filter before test"
            return False

        #check parameter
        if (x.nRow <> len(y)):
            print "ERROR!x.nRow should == len(y)"
            return False

        #stores new rows, cols, and vals
        newRows = [0]
        newCols = []
        newVals = []

        for r in range(x.nRow):
            curRowLen = 0

            #debug
            #print "===new doc==="

            for c in range(x.rows[r], x.rows[r + 1]):
                if self.idMap[x.cols[c]] >= 0 :
                    newCols.append(self.idMap[x.cols[c]])
                    newVals.append(x.vals[c])
                    curRowLen += 1

            newRows.append(newRows[len(newRows) - 1] + curRowLen)
        return [Matrix(newRows, newCols, newVals), y]

    """
    create a blackList by given x,y
    @rate is a percentage of selected feature
    using next formulation:
    X^2(t, c) =   N * (AD - CB)^2
                ____________________
                (A+C)(B+D)(A+B)(C+D)
    A,B,C,D is doc-count
    A:     belong to c,     include t
    B: Not belong to c,     include t
    C:     belong to c, Not include t
    D: Not belong to c, Not include t
    
    B = t's doc-count - A
    C = c's doc-count - A
    D = N - A - B - C

    and score of t can be calculated by next 2 formulations:
    X^2(t) = sigma p(ci)X^2(t,ci) (avg)
               i
    X^2(t) = max { X^2(t,c) }     (max)
    @return true if succeed
    """
    def TrainFilter(self, x, y):
        #check parameter
        if not ((self.method == "avg") or (self.method == "max")):
            print "ERROR!method should be avg or max"
            return False

        if (x.nRow <> len(y)):
            print "ERROR!x.nRow should == len(y)"
            return False

        #using y get set of target
        yy = set(y)
        yy = list(yy)
        yy.sort()

        #create a table stores X^2(t, c)
        #create a table stores A(belong to c, and include t
        chiTable = [[0 for i in range(x.nCol)] for j in range(yy[len(yy) - 1] + 1)]
        aTable = [[0 for i in range(x.nCol)] for j in range(yy[len(yy) - 1] + 1)]

        #calculate a-table
        for row in range(x.nRow):
            for col in range(x.rows[row], x.rows[row + 1]):
                aTable[y[row]][x.cols[col]] += 1

        #calculate chi-table
        n = x.nRow
        for t in range(x.nCol):
            for cc in range(len(yy)):
                #get a
                a = aTable[yy[cc]][t]
                #get b
                b = GlobalInfo.idToDocCount[t] - a
                #get c
                c = GlobalInfo.classToDocCount[yy[cc]] - a
                #get d
                d = n - a - b -c
                #get X^2(t, c)
                numberator = float(n) * (a*d - c*b) * (a*d - c*b)
                denominator = float(a+c) * (b+d) * (a+b) * (c+d)
                chiTable[yy[cc]][t] = numberator / denominator

        #calculate chi-score of each t
        #chiScore is [score, t's id] ...(n)
        chiScore = [[0 for i in range(2)] for j in range(x.nCol)]
        if (self.method == "avg"):
            #calculate prior prob of each c
            priorC = [0 for i in range(yy[len(yy) - 1] + 1)]
            for i in range(len(yy)):
                priorC[yy[i]] = float(GlobalInfo.classToDocCount[yy[i]]) / n

            #calculate score of each t
            for t in range(x.nCol):
                chiScore[t][1] = t
                for c in range(len(yy)):
                    chiScore[t][0] += priorC[yy[c]] * chiTable[yy[c]][t]
        else:
            #calculate score of each t
            for t in range(x.nCol):
                chiScore[t][1] = t
                for c in range(len(yy)):
                    if (chiScore[t][0] < chiTable[yy[c]][t]):
                        chiScore[t][0] = chiTable[yy[c]][t]

        #sort for chi-score, and make blackList
        chiScore = sorted(chiScore, key = lambda chiType:chiType[0], reverse = True)

        #init idmap
        self.idMap = [0 for i in range(x.nCol)]

        #add un-selected feature-id to idmap
        for i in range(int(self.rate * len(chiScore)), len(chiScore)):
            self.idMap[chiScore[i][1]] = -1
        offset = 0
        for i in range(x.nCol):
            if (self.idMap[i] < 0):
                offset += 1
            else:
                self.idMap[i] = i - offset
                GlobalInfo.newIdToId[i - offset] = i

        #output model information
        if (self.modelPath <> ""):
            f = open(self.modelPath, "w")
            modelStr = pickle.dumps([self.idMap], 1)
            pickle.dump(modelStr, f)
            f.close()

        #output chiSquare info
        if (self.logPath <> ""):
            f = open(self.logPath, "w")
            f.write("chiSquare info:\n")
            f.write("=======selected========\n")
            for i in range(len(chiScore)):
                if (i == int(self.rate * len(chiScore))):
                    f.write("========unselected=======\n")
                term = GlobalInfo.idToTerm[chiScore[i][1]]
                score = chiScore[i][0]
                f.write(term.encode("utf-8") + " " + str(score) + "\n")
            f.close()

        self.trained = True

        return True

"""
if __name__ == "__main__":
    config = Configuration.FromFile("conf/test.xml")
    GlobalInfo.Init(config, "__global__")
    txt2mat = Text2Matrix(config, "__matrix__")
    [trainx, trainy] = txt2mat.CreateTrainMatrix("data/tuangou_titles3.txt")
    chiFilter = ChiSquareFilter(config, "__filter__")
    chiFilter.TrainFilter(trainx, trainy) 
"""
