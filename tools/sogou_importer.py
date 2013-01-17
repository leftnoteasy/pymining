#encoding=utf-8

import dircache
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) <> 3:
        print "python sogou_importer.py sogou_path output_file_path"
        sys.exit()
    
    dirNameDict = { \
                   "C000007":0, \
                   "C000008":1, \
                   "C000010":2, \
                   "C000013":3, \
                   "C000014":4, \
                   "C000016":5, \
                   "C000020":6, \
                   "C000022":7, \
                   "C000023":8, \
                   "C000024":9, \
                  }
     
    outputPath = sys.argv[2]
    inputDir = sys.argv[1]
    ofs = open(outputPath, "w")

    if inputDir[len(inputDir) - 1] != "/":
        inputDir += "/"

    for dirName in dirNameDict.keys():
        newDir = inputDir + dirName + "/"
        if (not os.path.exists(newDir)):
            continue
        fileList = dircache.listdir(newDir)
        for fileName in fileList:
            filePath = newDir + fileName
            if (not os.path.exists(filePath)):
                continue
            ifs = open(filePath, "r")
            fileContent = ifs.read()
            fileContent = fileContent.decode("gb18030", "ignore")
            fileContent = fileContent.replace("\n", " ")
            fileContent = fileContent.replace("\t", " ")
            ofs.write(fileContent.encode("utf-8") + "\t" + str(dirNameDict[dirName]) + "\n")
            ifs.close()

    ofs.close()
