import math
from ..nlp.segmenter import Segmenter
from ..math.matrix import Matrix
from ..common.global_info import GlobalInfo
from ..common.configuration import Configuration

class Text2Matrix:
    def __init__(self, config, nodeName, loadFromFile = False):
        self.node = config.GetChild(nodeName)
        self.segmenter = Segmenter(config, "__segmenter__")
        self.trained = loadFromFile
        GlobalInfo.Init(config, "__global__", loadFromFile)
        
    """
    create train matrix:
        fill dict in GlobalInfo, record:
        1)termToId
        2)idToTerm
        3)termToDocCount
        4)classToDocCount
        and save mat-x using csr, save mat-y using list
    """
    def CreateTrainMatrix(self, path = ""):
        #get input-path
        inputPath = path
        if (inputPath == ""):
            inputPath = self.node.GetChild("train_input").GetValue()

        f = open(inputPath, "r")
        uid = 0
        rows = [0]
        cols = []
        vals = []
        y = []

        #fill the matrix's cols and rows
        for line in f:
            vec = line.split("\t")
            line = vec[0]
            target = int(vec[1])
            y.append(target)
            wordList = self.segmenter.Split(line.decode("utf-8"))

            #store current row's cols
            partCols = []

            #create dicts and fill partCol
            #calculate term-frequent in this loop
            curWordCount = 0
            termFres = {}
            for word in wordList:
                curWordCount += 1
                if (not GlobalInfo.termToId.has_key(word)):
                    GlobalInfo.termToId[word] = uid
                    GlobalInfo.idToTerm[uid] = word
                    uid += 1
                termId = GlobalInfo.termToId[word]
                partCols.append(termId)
                if (not termFres.has_key(termId)):
                    termFres[termId] = 1
                else:
                    termFres[termId] += 1
            #fill partCol
            partCols = set(partCols)
            partCols = list(partCols)
            partCols.sort()

            #fill cols and vals, fill termToDocCount
            for col in partCols:
                cols.append(col)
                #fill vals with termFrequent
                vals.append(termFres[col])
                #fill idToDocCount
                if (not GlobalInfo.idToDocCount.has_key(col)):
                    GlobalInfo.idToDocCount[col] = 1
                else:
                    GlobalInfo.idToDocCount[col] += 1

            #fill rows
            rows.append(rows[len(rows) - 1] + \
                len(partCols))

            #fill classToDocCount
            if (not GlobalInfo.classToDocCount.has_key(target)):
                GlobalInfo.classToDocCount[target] = 1
            else:
                GlobalInfo.classToDocCount[target] += 1

        #fill GlobalInfo's idToIdf
        for termId in GlobalInfo.idToTerm.keys():
            GlobalInfo.idToIdf[termId] = math.log(float(len(rows) - 1) / (GlobalInfo.idToDocCount[termId] + 1))

        #NOTE: now, not mul idf to vals, because not all algorithms need tf * idf
        #change matrix's vals using tf-idf represent
        #for r in range(len(rows) - 1):
        #    for c in range(rows[r], rows[r + 1]):
        #        termId = cols[c]
        #        #idf(i) = log(|D| / |{d (ti included)}| + 1
        #        vals[c] = vals[c] * GlobalInfo.idToIdf[termId]

        #close file
        f.close()

        #write dicts out
        GlobalInfo.Write()

        self.trained = True

        return [Matrix(rows, cols, vals), y] 

    def CreatePredictSample(self, src):
        print src
        if (not self.trained):
            print "train Classifier Matrix before predict"
        
        #split sentence
        #if src is read from utf-8 file directly, 
        #    should using CreatePredictSample(src.decode("utf-8"))
        wordList = self.segmenter.Split(src)
        cols = []
        vals = []
        #fill partCols, and create csr
        partCols = []
        termFreqs = {}
        for word in wordList:
            if (GlobalInfo.termToId.has_key(word)):
                termId = GlobalInfo.termToId[word]
                partCols.append(termId)
                if (not termFreqs.has_key(termId)):
                    termFreqs[termId] = 1
                else:
                    termFreqs[termId] += 1
        partCols = set(partCols)
        partCols = list(partCols)
        partCols.sort()
        for col in partCols:
            cols.append(col)
            vals.append(termFreqs[col])

        return [cols, vals]

    """
    create predict matrix using previous dict
    """
    def CreatePredictMatrix(self, path = ""):
        if (not self.trained):
            print "train ClassifierMatrix before predict"
            return False

        #get input path
        inputPath = path
        if (inputPath == ""):
            inputPath = self.curNode.GetChild("test_input")

        f = open(inputPath, "r")
        rows = [0]
        cols = []
        vals = []
        y = []
        for line in f:
            vec = line.split("\t")
            line = vec[0]
            y.append(int(vec[1]))

            #split sentence
            wordList = self.segmenter.Split(line.decode("utf-8"))

            #fill partCols, and create csr
            partCols = []
            termFreqs = {}
            curWordCount = 0
            for word in wordList:
                curWordCount += 1
                if (GlobalInfo.termToId.has_key(word)):
                    termId = GlobalInfo.termToId[word]
                    partCols.append(termId)
                    if (not termFreqs.has_key(termId)):
                        termFreqs[termId] = 1
                    else:
                        termFreqs[termId] += 1

            partCols = set(partCols)
            partCols = list(partCols)
            partCols.sort()
            for col in partCols:
                cols.append(col)
                vals.append(termFreqs[col])
            rows.append(rows[len(rows) - 1] + \
                    len(partCols))

        #close file
        f.close()
        return [Matrix(rows, cols, vals), y]

if __name__ == "__main__":
    config = Configuration.FromFile("conf/test.xml")
    txt2mat = Text2Matrix(config, "__matrix__")
    [trainMat, ty] = txt2mat.CreateTrainMatrix("data/tuangou_titles3.txt")
    [predictMat, py] = txt2mat.CreatePredictMatrix("data/tuangou_titles3.txt")
    print py
    print predictMat.rows
    print predictMat.cols
    print predictMat.vals
