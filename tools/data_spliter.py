"""
split a import data to train-set and test-set randomly

USAGE:
python data_spliter.py input_data rate_for_train train_file test_file

EXAMPLE:
python data_spliter.py sogou.txt 0.3 train.txt test.txt
"""

import sys
import random

if __name__ == "__main__":
    if (len(sys.argv) != 5):
        print "usage: python data_spliter.py input_data rate_for_train train_file test_file"
        sys.exit()

    inputFile = sys.argv[1]
    rate = float(sys.argv[2])
    trainOut = sys.argv[3]
    testOut = sys.argv[4]

    fin = open(inputFile, "r")
    fTrain = open(trainOut, "w")
    fTest = open(testOut, "w")
    for line in fin:
        curRandom = random.random()
        if (curRandom <= rate):
            fTrain.write(line)
        else:
            fTest.write(line)
    
    fin.close()
    fTrain.close()
    fTest.close()
