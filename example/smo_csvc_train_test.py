import sys, os
import math
sys.path.append(os.path.join(os.getcwd(), '../'))

from pymining.math.matrix import Matrix
from pymining.math.text2matrix import Text2Matrix
from pymining.nlp.segmenter import Segmenter
from pymining.common.global_info import GlobalInfo
from pymining.common.configuration import Configuration
from pymining.preprocessor.chisquare_filter import ChiSquareFilter
from pymining.classifier.smo_csvc import *
from pymining.preprocessor.data_format import *
import time
from time import clock as now

if __name__ == "__main__": 
    config = Configuration.FromFile("conf/test.xml")
#---------------------------------------------------------------------------------1.text data-----------------------------------------------------------------------------------
### C=100 kernel=RBF p=0.03
### Recall =  0.973913043478 Precision =  0.888888888889 Accuracy =  0.886666666667 
### F(beta=1) =  0.604792440094 F(beta=2) =  0.738299274885 AUCb =  0.786956521739
    GlobalInfo.Init(config, "__global__")
    txt2mat = Text2Matrix(config, "__matrix__")
    [trainx, trainy] = txt2mat.CreateTrainMatrix("data/train.txt")
    chiFilter = ChiSquareFilter(config, "__filter__")
    chiFilter.TrainFilter(trainx, trainy)
    [trainx, trainy] = chiFilter.MatrixFilter(trainx, trainy)

    for i in range(0,trainx.nRow):
        if trainy[i] == 3:
            trainy[i] = 1
        else:
            trainy[i] = -1

    [testx, testy] = txt2mat.CreatePredictMatrix("data/test.txt")   
    [testx, testy] = chiFilter.MatrixFilter(testx, testy)
    testx.nCol = trainx.nCol
    for i in range(0,testx.nRow):
        if testy[i] == 3:
            testy[i] = 1
        else:
            testy[i] = -1

    nbModel = Smo_Csvc(config, 'smo_csvc',False,False,1)
    [duration,iterations] = nbModel.Train(trainx, trainy) 
    print 'duration of SMO training is:',duration,' and iterations of it is :',iterations
    print trainx.nRow,nbModel.model.svn
    nbModel = Smo_Csvc(config, 'smo_csvc', True)
    [Recall,Precision,Accuracy,Fbeta1,Fbeta2,AUCb,auc] = nbModel.Test(testx, testy)
#---------------------------------------------------------------------------------text data-----------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------2.SPECT data-----------------------------------------------------------------------------------
### C = 100,kernel=RBF,gama = 15
### Recall =  0.93023255814 Precision =  0.958083832335 Accuracy =  0.898395721925 
### F(beta=1) =  0.617135142954 F(beta=2) =  0.756787437329 AUCb =  0.731782945736

    [trainx,trainy] = Data_Format.data2dense_matrix("data/spectf/SPECT.train.txt","data/spectf/SPECT.train.txt",',',',')
    nbModel = Smo_Csvc(config, 'smo_csvc',False,False,1)
    nbModel.model.config.isdense = True
    [duration,iterates] = nbModel.Train(trainx, trainy)
    print 'duration = ', duration,'iterates = ',iterates

    [testx,testy] = Data_Format.data2dense_matrix("data/spectf/SPECT.test.txt","data/spectf/SPECT.test.txt",',',',')
    nbModel = Smo_Csvc(config, 'smo_csvc', True)
    nbModel.model.config.isdense = True
    [Recall,Precision,Accuracy,Fbeta1,Fbeta2,AUCb,auc] = nbModel.Test(testx, testy)
#---------------------------------------------------------------------------------SPECT data-----------------------------------------------------------------------------------

#---------------------------------------------------------------------------------3.arece data-----------------------------------------------------------------------------------
### C = 100,kernel = RBF,gama = 0.0000000011
### Recall =  0.772727272727 Precision =  0.85 Accuracy =  0.84 
### F(beta=1) =  0.500866551127 F(beta=2) =  0.584074373484 AUCb =  0.832792207792

    [trainx,trainy] = Data_Format.data2dense_matrix("data/arcene/arcene_train.data","data/arcene/arcene_train.labels",' ','\n')
    nbModel = Smo_Csvc(config, 'smo_csvc',False,False,0.5)
    nbModel.model.config.isdense = True
    [duration,iterates] = nbModel.Train(trainx, trainy)
    print 'duration = ', duration,'iterates = ',iterates

    [testx,testy] = Data_Format.data2dense_matrix("data/arcene/arcene_valid.data","data/arcene/arcene_valid.labels",' ','\n')
    nbModel = Smo_Csvc(config, 'smo_csvc', True)
    nbModel.model.config.isdense = True
    [Recall,Precision,Accuracy,Fbeta1,Fbeta2,AUCb,auc] = nbModel.Test(testx, testy)    
#---------------------------------------------------------------------------------arece data-----------------------------------------------------------------------------------

#---------------------------------------------------------------------------------4.dexter data-----------------------------------------------------------------------------------
### C=100 kernel = RBF p = 0.000001   
### Recall =  0.986666666667 Precision =  0.822222222222 Accuracy =  0.886666666667 
### F(beta=1) =  0.577637130802 F(beta=2) =  0.698291252232 AUCb =  0.886666666667 
 
    rows = []
    cols = []
    vals = []
    y = []
    ir = 0
    f = open("data/dexter/dexter_train.data", 'r')
    for each in f:
        each= each.split(' ')
        l = len(each) - 1
        rows.append(ir)
        for i in range(0, l):
            c = each[i].split(':')   
            cols.append(int(c[0]))
            vals.append(float(c[1]))
            ir += 1        
    rows.append(len(vals))
        
    trainx = Matrix(rows,cols,vals,len(rows)-1,20000)  
    
    f = open("data/dexter/dexter_train.labels", 'r')
    for each in f:
        each= each.split('\n') 
        y.append(float(each[0]))
        
    trainy = y
    
    nbModel = Smo_Csvc(config, 'smo_csvc')
    nbModel.model.config.isdense = False
    
    start = now()
    nbModel.Train(trainx, trainy)
    finish = now()
    print 'elapsed time:', finish - start   

    rows = []
    cols = []
    vals = []
    y = []
    ir = 0
    f = open("data/dexter/dexter_valid.data", 'r')
    for each in f:
        each= each.split(' ')
        l = len(each) - 1
        rows.append(ir)
        for i in range(0, l):
            c = each[i].split(':')   
            cols.append(int(c[0]))
            vals.append(float(c[1]))
            ir += 1        
    rows.append(len(vals))
        
    trainx = Matrix(rows,cols,vals,len(rows)-1,20000)  
    
    
    nbModel = Smo_Csvc(config, 'smo_csvc', True)
    
    testx = Matrix(rows,cols,vals,len(rows)-1,20000)
        
    f = open("data/dexter/dexter_valid.labels", 'r')
    for each in f:
        each= each.split('\n')             
        y.append(float(each[0]))
        
    testy = y
    nbModel.model.config.isdense = False
   
    nbModel.Test(testx, testy)
#---------------------------------------------------------------------------------dexter data-----------------------------------------------------------------------------------

#---------------------------------------------------------------------------------5.gisette data-----------------------------------------------------------------------------------
### C=100 kernel=RBF p = 0.00000000005
### Recall =  0.972 Precision =  0.983805668016 Accuracy =  0.978
### F(beta=1) =  0.647037875094 F(beta=2) =  0.802795761493 AUCb =  0.978
    [trainx,trainy] = Data_Format.data2sparse_matrix("data/gisette/gisette_train.data","data/gisette/gisette_train.labels",5000,' ','\n')
    nbModel = Smo_Csvc(config, 'smo_csvc',False,False,1)
    [duration,iterates] = nbModel.Train(trainx, trainy)
    print 'duration = ', duration,'iterates = ',iterates

    [testx,testy] = Data_Format.data2sparse_matrix("data/gisette/gisette_valid.data","data/gisette/gisette_valid.labels",5000,' ','\n')
    nbModel = Smo_Csvc(config, 'smo_csvc', True)
    [Recall,Precision,Accuracy,Fbeta1,Fbeta2,AUCb,auc] = nbModel.Test(testx, testy)   

#---------------------------------------------------------------------------------gisette data-----------------------------------------------------------------------------------

#---------------------------------------------------------------------------------6.mushrooms data-----------------------------------------------------------------------------------
### C=100 kernel=RBF
### Recall =  1.0 Precision =  0.942615239887 Accuracy =  0.960897435897 
### F(beta=1) =  0.640664961637 F(beta=2) =  0.793097989552 AUCb =  0.945340501792
    [trainx,trainy] = Data_Format.data2libsparse_matrix("data/mushrooms/mushrooms_train.data","data/mushrooms/mushrooms_train.data",112,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc',False,False) 
    [duration,iterates] = nbModel.Train(trainx, trainy)
    print 'duration = ', duration,'iterates = ',iterates

    [testx,testy] = Data_Format.data2libsparse_matrix("data/mushrooms/mushrooms_valid.data","data/mushrooms/mushrooms_valid.data",112,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc', True)
    [Recall,Precision,Accuracy,Fbeta1,Fbeta2,AUCb,auc] = nbModel.Test(testx, testy)
#---------------------------------------------------------------------------------mushrooms data-----------------------------------------------------------------------------------

#---------------------------------------------------------------------------------7.madelon data-----------------------------------------------------------------------------------
## C=100 kernel = RBF p = 0.01 
## Recall =  0.673333333333 Precision =  0.724014336918 Accuracy =  0.708333333333 
## F(beta=1) =  0.406701950583 F(beta=2) =  0.451613474471 AUCb =  0.708333333333
    [trainx,trainy] = Data_Format.data2dense_matrix("data/madelon/madelon_train.data","data/madelon/madelon_train.labels",' ',' ')
    nbModel = Smo_Csvc(config, 'smo_csvc',False,False)
    nbModel.model.config.isdense = True
    [duration,iterates] = nbModel.Train(trainx, trainy)
    print 'duration = ', duration,'iterates = ',iterates
    [testx,testy] = Data_Format.data2dense_matrix("data/madelon/madelon_valid.data","data/madelon/madelon_valid.labels",' ',' ')
    nbModel = Smo_Csvc(config, 'smo_csvc', True)  
    nbModel.model.config.isdense = True
    [Recall,Precision,Accuracy,Fbeta1,Fbeta2,AUCb,auc] = nbModel.Test(testx, testy)  
 
#---------------------------------------------------------------------------------madelon data-----------------------------------------------------------------------------------

#---------------------------------------------------------------------------------8.adult data-----------------------------------------------------------------------------------
### C=100 kernel = RBF p = 0.08
### Recall =  0.598710717164 Precision =  0.588281868567 Accuracy =  0.802687685748 
### F(beta=1) =  0.322095887953 F(beta=2) =  0.339513363093 AUCb =  0.733000615919
    [trainx,trainy] = Data_Format.data2libsparse_matrix("data/adult/adult_train.data","data/adult/adult_train.data",123,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc',False,False,10)
    [duration,iterates] = nbModel.Train(trainx, trainy)
    print 'duration = ', duration,'iterates = ',iterates

    [testx,testy] = Data_Format.data2libsparse_matrix("data/adult/adult_valid.data","data/adult/adult_valid.data",123,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc', True)
    [Recall,Precision,Accuracy,Fbeta1,Fbeta2,AUCb,auc] = nbModel.Test(testx, testy)   
#---------------------------------------------------------------------------------adult data----------------------------------------------------------------------------------- 

#---------------------------------------------------------------------------------9.fourclass data-----------------------------------------------------------------------------------
### C=100 kernel = RBF p = 0.01
### Recall =  0.991071428571 Precision =  1.0 Accuracy =  0.996415770609 
### F(beta=1) =  0.662686567164 F(beta=2) =  0.827123695976 AUCb =  0.995535714286
    [trainx,trainy] = Data_Format.data2libsparse_matrix("data/fourclass/fourclass_train.data","data/fourclass/fourclass_train.data",2,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc')
    [duration,iterates] = nbModel.Train(trainx, trainy)
    print 'duration = ', duration,'iterates = ',iterates

    [testx,testy] = Data_Format.data2libsparse_matrix("data/fourclass/fourclass_valid.data","data/fourclass/fourclass_valid.data",2,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc', True)
    [Recall,Precision,Accuracy,Fbeta1,Fbeta2,AUCb,auc] = nbModel.Test(testx, testy)  
#---------------------------------------------------------------------------------fourclass data-----------------------------------------------------------------------------------    
    
    
#--------------------------------------------------10.australian data-----------------------------------------------------------------------------------
### C=100 kernel = RBF p = 0.0001 150
### Recall =  0.801886792453 Precision =  0.643939393939 Accuracy =  0.714285714286 
### F(beta=1) =  0.422243001578 F(beta=2) =  0.474093808236 AUCb =  0.722913093196
    [trainx,trainy] = Data_Format.data2libsparse_matrix("data/australian/australian_train.data","data/australian/australian_train.data",14,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc',False,False,150)
    [duration,iterates] = nbModel.Train(trainx, trainy)
    print 'duration = ', duration,'iterates = ',iterates

    [testx,testy] = Data_Format.data2libsparse_matrix("data/australian/australian_valid.data","data/australian/australian_valid.data",14,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc', True)  
    [Recall,Precision,Accuracy,Fbeta1,Fbeta2,AUCb,auc] = nbModel.Test(testx, testy)   
#---------------------------------------------------------------------------------australian data-----------------------------------------------------------------------------------

#--------------------------------------------------11.splice data-----------------------------------------------------------------------------------
### C=100 kernel = RBF p = 0.01 
### Recall =  0.885057471264 Precision =  0.912488605287 Accuracy =  0.896091954023 
### F(beta=1) =  0.577366617352 F(beta=2) =  0.696505768897 AUCb =  0.896551724138
    [trainx,trainy] = Data_Format.data2libsparse_matrix("data/splice/splice_train.data","data/splice/splice_train.data",60,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc',False,False)
    [duration,iterates] = nbModel.Train(trainx, trainy)
    print 'duration = ', duration,'iterates = ',iterates

    [testx,testy] = Data_Format.data2libsparse_matrix("data/splice/splice_valid.data","data/splice/splice_valid.data",60,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc', True)  
    [Recall,Precision,Accuracy,Fbeta1,Fbeta2,AUCb,auc] = nbModel.Test(testx, testy)   
#---------------------------------------------------------------------------------splice data-----------------------------------------------------------------------------------

#--------------------------------------------------12.svmguide1 data-----------------------------------------------------------------------------------
### C=100 kernel = RBF p = 0.01 
### Recall =  0.964 Precision =  0.957774465971 Accuracy =  0.96075 
### F(beta=1) =  0.632009483243 F(beta=2) =  0.7795759451 AUCb =  0.96075
    [trainx,trainy] = Data_Format.data2libsparse_matrix("data/svmguide1/svmguide1_train.data","data/svmguide1/svmguide1_train.data",4,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc',False,False)
    [duration,iterates] = nbModel.Train(trainx, trainy)
    print 'duration = ', duration,'iterates = ',iterates

    [testx,testy] = Data_Format.data2libsparse_matrix("data/svmguide1/svmguide1_valid.data","data/svmguide1/svmguide1_valid.data",4,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc', True)  
    [Recall,Precision,Accuracy,Fbeta1,Fbeta2,AUCb,auc] = nbModel.Test(testx, testy)   
#---------------------------------------------------------------------------------svmguide1 data-----------------------------------------------------------------------------------

#--------------------------------------------------13.rcv1 data-----------------------------------------------------------------------------------
## C=100 kernel = RBF p = 0.01 
## Recall =  0.956610366919 Precision =  0.964097045588 Accuracy =  0.9593  
## F(beta=1) =  0.631535514017 F(beta=2) =  0.778847158177 AUCb =  0.959383756361
    [trainx,trainy] = Data_Format.data2libsparse_matrix("data/rcv1/rcv1_train.data","data/rcv1/rcv1_train.data",47236,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc',False,False)
    [duration,iterates] = nbModel.Train(trainx, trainy)
    print 'duration = ', duration,'iterates = ',iterates

    [testx,testy] = Data_Format.data2libsparse_matrix("data/rcv1/rcv1_valid.data","data/rcv1/rcv1_valid.data",47236,' ',' ',':')
    nbModel = Smo_Csvc(config, 'smo_csvc', True)  
    [Recall,Precision,Accuracy,Fbeta1,Fbeta2,AUCb,auc] = nbModel.Test(testx, testy) 

#---------------------------------------------------------------------------------rcv1 data-----------------------------------------------------------------------------------
