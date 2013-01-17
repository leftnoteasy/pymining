import math
import numpy

from ..math.matrix import Matrix
from numpy import *

class Data_Format:
    @staticmethod
    def self_scale_data(sparam, trainx):
        '''to scale data with itself.[-1,1]'''
        
        if sparam.isdense == True:
            for i in range(0,trainx.shape[0]):
                for j in range(0,trainx.shape[1]):
                    trainx[i,j] = trainx[i,j]/(sqrt(trainx[i,j]*trainx[i,j]) + 10)
        else:
            for i in range(0,len(trainx.vals)):
                trainx.vals[i] = trainx.vals[i]/(sqrt(trainx.vals[i]*trainx.vals[i]) + 10)
                
    @staticmethod
    def data2sparse_matrix(data_path,label_path,feature_number,separator1,separator2):
        cur = 0
        rows = []
        cols = []
        vals = []
        y = []
        count = 0
        if data_path == label_path:
            start = 1
        else:
            start = 0
        f = open(data_path, 'r')
        for each in f:
            print 'to deal with sample ',count
            count += 1            
            each= each.split(separator1)
            l = len(each) - 1
            
            if data_path == label_path:
                y.append(int(each[0]))
                
            rows.append(cur)
            for i in range(start, l):
                if float(each[i]) != 0:
                    cols.append(i)
                    vals.append(float(each[i]))
                    cur += 1
        rows.append(len(vals))   
        trainx = Matrix(rows,cols,vals,len(rows)-1,feature_number) 
        
        if data_path != label_path:            
            f = open(label_path, 'r')
            count = 0
            for each in f:
                count += 1                
                each= each.split(separator2)             
                y.append(int(each[0]))                
        trainy = y    
        return [trainx,trainy]    
   
    @staticmethod
    def data2libsparse_matrix(data_path,label_path,feature_number,separator1,separator2,separator3):
        cur = 0
        rows = []
        cols = []
        vals = []
        y = []
        count = 0
        if data_path == label_path:
            start = 1
        else:
            start = 0
        f = open(data_path, 'r')
        for each in f:
            print 'to deal with sample ',count
            count += 1            
            each= each.split(separator1)
            l = len(each) - 1
            
            if data_path == label_path:
                y.append(int(each[0]))
                
            rows.append(cur)
            for i in range(start, l):
                    c = each[i].split(separator3)   
                    cols.append(int(c[0]))
                    vals.append(float(c[1]))
                    cur += 1
        rows.append(len(vals))   
        trainx = Matrix(rows,cols,vals,len(rows)-1,feature_number) 
        
        if data_path != label_path:            
            f = open(label_path, 'r')
            count = 0
            for each in f:
                count += 1                
                each= each.split(separator2)             
                y.append(int(each[0]))                
        trainy = y    
        return [trainx,trainy] 
     
    @staticmethod
    def data2dense_matrix(data_path,label_path,separator1,separator2):
        
        x = []
        y = []
        if data_path == label_path:
            start = 1
        else:
            start = 0  
                        
        f = open(data_path, 'r')
        for each in f:
            each= each.split(separator1)
            l = len(each) - 1
            d = []
            for i in range(start, l):
                d.append(float(each[i]))
            x.append(d)
            
            if data_path == label_path:
                if int(each[0]) == 0:
                    y.append(-1)
                else:
                    y.append(int(each[0]))                    
        trainx = matrix(x)
        
        if data_path != label_path:
            f = open(label_path, 'r')
            for each in f:
                each= each.split(separator2) 
                y.append(int(each[0]))            
        trainy = y
        
        return [trainx,trainy]    
    
    @staticmethod
    def get_variance (list):
        '''to get variance of a list.'''
        
        meanvalue = sum(list) * 1.0 / len(list)        
        return sum([(element - meanvalue) ** 2 for element in list]) / len(list)
    
    @staticmethod
    def max_min_scale_data(sparam, trainx):
        '''using max-min method to scale data.[0,1]'''
        
        if sparam.isdense == True:
            for i in range(0,trainx.shape[1]):
                minvalue = min(trainx[:,i])
                maxvalue = max(trainx[:,i])
                for j in range(0,trainx.shape[0]):
                    if maxvalue - minvalue == 0:
                        trainx[j,i] = 0
                    else:
                        trainx[j,i] = float(float(trainx[j,i] - minvalue)/float(maxvalue - minvalue)) 
        else:
            mindic = {}
            maxdic = {}
            for i in range(0,len(trainx.cols)):
                key = trainx.cols[i]
                if mindic.has_key(key):
                    if float(mindic[key]) > trainx.vals[i]:
                        mindic[key] = trainx.vals[i]
                        continue
                else:
                    mindic[key] = trainx.vals[i]
                 
                key = trainx.cols[i]    
                if maxdic.has_key(key):
                    if float(maxdic[key]) < trainx.vals[i]:
                        maxdic[key] = trainx.vals[i]
                        continue
                else:
                    maxdic[key] = trainx.vals[i]
                    
            for j in range(0,len(trainx.vals)):
                key = trainx.cols[j]
                if float(maxdic[key]) - float(mindic[key]) == 0:
                    trainx.vals[j] = 0
                else:
                    trainx.vals[j] = (trainx.vals[j] - float(mindic[key]))/(float(maxdic[key]) - float(mindic[key]))
    
    @staticmethod
    def average_variance_scale_data(sparam, trainx):
        '''using average variance method to scale data,[-1,1]'''
        
        if sparam.isdense == True:
            for i in range(0,trainx.shape[1]):
                meanvalue = sum(trainx[:,i])/trainx.shape[0]
                varvalue = Data_Format.get_variance(trainx[:,i])
                for j in range(0,trainx.shape[0]):
                    if varvalue == 0:
                        trainx[j,i] = 0
                    else:
                        trainx[j,i] = float(float(trainx[j,i] - meanvalue)/varvalue) 
        else:
            mindic = {}
            maxdic = {}
            for i in range(0,len(trainx.cols)):
                key = trainx.cols[i]
                if mindic.has_key(key):
                    lst = mindic[key]
                    lst.append(trainx.vals[i])
                else:
                    lst = []
                    lst.append(trainx.vals[i])
                mindic[key] = lst                
                    
            for j in range(0,len(trainx.vals)):
                key = trainx.cols[j]
                meanvalue = sum(mindic[key])/len(mindic[key])
                varvalue = Data_Format.get_variance(mindic[key])
                if varvalue == 0:
                    trainx.vals[j] = 0
                else:
                    trainx.vals[j] = float(float(trainx.vals[j] - meanvalue)/varvalue) 
    