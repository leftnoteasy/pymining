#coding=utf8=
#mmseg
import matrix
import random_forest
from matrix import *
from random_forest import *
def Segment(line, dicts):
    index = 0
    wordList = []
    while index < len(line):
        finded = False
        #for i = 4 -> 2
        for i in range(2, 4, 1) [::-1]:
            if (i + index <= len(line)):
                if (dicts.has_key(line[index : i + index])):
                        wordList.append(line[index : i + index])
                        index += i
                        finded = True
                        break
        if (finded):
            continue
        index += 1
        
    return wordList

#load dict from a specified path
def LoadDict(dictPath):
    f = open(dictPath, "r")
    dicts = {}
    stopwords = {}
    for line in f:
        line = line.decode("utf-8")
        l = line.split()
        dicts[l[1]] = 1            
    f.close()
    return dicts

#process data file
def CreateMatrixFromData(dataPath, srcDict):
    #define csr row, col, val
    rows = [0]
    cols = []
    vals = []
    #define target num of each sample
    tars = []
    
    #2. open data file, create tarDict from samples
    tarDict = {}
    tarDictNum = {}
    f = open(dataPath, "r")
    uid = 0
    for line in f:
        #line format as: content \t class
        line = line.decode("utf-8")
        vec = line.split("\t")
        tars.append(int(vec[1]))
        
        words = Segment(line, srcDict)
        for word in words:
            if (not tarDict.has_key(word)):
                tarDict[word] = uid
                tarDictNum[word] = 0
                uid += 1
            else:
                tarDictNum[word] += 1

    for word in tarDictNum:
        if (tarDictNum[word] >= 20):
            del tarDict[word]
                
    #3. re-open data file, using tarDict, and create csr
    f.close()
    f = open(dataPath, "r")
    for line in f:
        words = Segment(line.decode("utf-8"), srcDict)
        partCols = []
        for word in words:
            if (tarDict.has_key(word)):
                partCols.append(tarDict[word])
        partCols = set(partCols)
        partCols = list(partCols)
        partCols.sort()
        for col in partCols:
            cols.append(col)
            vals.append(1)
        rows.append(rows[len(rows) - 1] + len(partCols))
    return [rows, cols, vals, tars, tarDict]
    
def Train(dataPath, nTree, ratio, srcDict):
    #using training data get mat-x and mat-y
    #[rows, cols, vals, y, tarDict] = CreateMatrixFromData("e:\src-code\python_platform\data\\tuangou_titles.txt")
    [rows, cols, vals, y, tarDict] = CreateMatrixFromData(dataPath, srcDict)
    x = Matrix(rows, cols, vals)
    model = RandomForest(x, y, nTree, ratio)
    model.Learn()
    return [model, tarDict]

def Predict(line, model, srcDict, tarDict):
    words = Segment(line, srcDict)
    partCols = []
    for word in words:
        if (tarDict.has_key(word)):
            partCols.append(tarDict[word])
    partCols = set(partCols)
    partCols = list(partCols)
    partCols.sort()

    rows = [0]
    cols = []
    vals = []
    for col in partCols:
        cols.append(col)
        vals.append(1)
    rows.append(len(partCols))

    print rows
    print cols
    return model.Predict(Matrix(rows, cols, vals))

if __name__ == "__main__":
    srcDict = LoadDict("dict/baidu_dict.txt")
    [model, tarDict] = Train("data/tuangou_titles_with_tag.txt", 1, 0.95, srcDict)
   
    print "after train"

    #using test data one by one
    correct = 0
    incorrect = 0
    f = open("data/tuangou_titles_with_tag.txt", "r")
    for line in f:
        line = line.decode("utf-8")
        vec = line.split("\t")
        predictResult = Predict(vec[0], model, srcDict, tarDict)
        tag = int(vec[1])
        if (tag == predictResult):
            correct += 1
        else:
            incorrect += 1
        print "source:", tag, " predicted:", predictResult

    print "precision:", correct * 1.00 / (correct + incorrect)


