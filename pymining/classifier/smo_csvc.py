import math
import matplotlib
import matplotlib.pyplot as plt
import numpy
import os
import pickle
import psyco
psyco.full()
import sys
import time

from ..common.global_info import GlobalInfo
from ..common.configuration import Configuration
from ..math.matrix import Matrix
from ..math.text2matrix import Text2Matrix
from numpy import *
from operator import itemgetter


class Svm_Param:
    '''the parameter configuration of csvc.'''
    
    def __init__(self,config, nodeName):        
        try:
            self.curNode = config.GetChild(nodeName)
            
            #-------------------begin model info-------------------------------
            self.modelpath = self.curNode.GetChild("model_path").GetValue()
            self.logpath = self.curNode.GetChild("log_path").GetValue()
            #store penalty coefficient of slack variable.
            self.C = float(self.curNode.GetChild("c").GetValue())
            #store a number nearly zero.
            self.eps = float(self.curNode.GetChild("eps").GetValue())
            #store tolerance of KKT conditions.
            self.tolerance = float(self.curNode.GetChild("tolerance").GetValue())
            #-------------------end  model info-------------------------------
            
            #-------------------begin  times info-------------------------------
            #log frequency 
            self.times = int(self.curNode.GetChild("times").GetValue())
            #-------------------end    times info-------------------------------
            
            #-------------------begin kernel info-------------------------------
            self.kernelnode = self.curNode.GetChild("kernel")
            
            #to get kernel's type. 
            self.kernel_type = self.kernelnode.GetChild("name").GetValue();
            #to get parameters from top to button -> from left to right -> from inner to outer.        
            self.parameters = self.kernelnode.GetChild("parameters").GetValue().split(',')     
            #-------------------end  kernel info-------------------------------
            #to get size of cache.
            self.cachesize = float(self.curNode.GetChild("cachesize").GetValue())
            #matrix is dense or sparse.
            self.isdense = False
        except Exception as detail:
            print 'to read configuration file error,detail is:', detail
         
class Svm_Model:
    '''the support vector machine model.'''
    
    def __init__(self,config, nodeName):
        #the configuration of svc.
        self.config = Svm_Param(config, nodeName)
        #the number of support vector machines.
        self.svn = 0
        #alpha
        self.alpha = []
        #the support vector.
        self.sv = None
        #the label of sv.
        self.label = []
        #the weight of model.
        self.w = []
        #the bias of model.
        self.b = 0.0
        
class Svm_Util:
    
    '''utilities of support vector machine.'''
    
    @staticmethod   
    def dot( sparam, trainx, trainy, i, j):
        
        '''to calculate dot product of two dense matrix or sparse matrix.'''
        
        if trainx == None or trainy == None:
            print 'train matrix should not be empty.'
            return -1
        if i < 0 or j < 0:
            print 'index must bigger then zero.'
            return -1
        try:
            #isdense = True -> dense matrix,isdense = False -> sparse matrix,
            isdense = sparam.isdense
            #the training set or sv is sparse matrix.
            if isdense == False:
               
                if trainx.nCol <> trainy.nCol:
                    print "the dimension of trainx and trainy must be equal. "    
                    return -1
                if i >= trainx.nRow or j >= trainy.nRow:
                    print "index i and j out. "    
                    return -1
                sum = 0.0
                
#                to calculate dot product with O(nlgn)
                i1 = trainx.rows[i]
                i2 = trainy.rows[j]
                p1 = 0  #the elements number of row i
                p2 = 0  #the elements number of row j
                if i < len(trainx.rows)-1 :
                    p1 = trainx.rows[i+1] - trainx.rows[i]
                
                if j < len(trainy.rows)-1 :
                    p2 = trainy.rows[j+1] - trainy.rows[j]                   
                 
                if p2 <= p1:     
                    curlow = i1      
                    for k in range(i2, i2+p2):  
                        pos = Svm_Util.binary_search(trainx.cols,curlow,i1+p1-1,trainy.cols[k])
                        if  pos != -1:
                            sum += trainy.vals[k] * trainx.vals[pos]
                            curlow = pos + 1
                else: 
                    curlow = i2
                    for k in range(i1, i1+p1):                     
                        pos = Svm_Util.binary_search(trainx.cols,curlow,i2+p2-1,trainx.cols[k])
                        if  pos != -1:
                            sum += trainx.vals[k] * trainy.vals[pos]
                            curlow = pos + 1 
                return sum
                
            else:
                if i >= trainx.shape[0] or j >= trainy.shape[0]:
                    print "index i or j out. "    
                    return -1                
                if trainx.ndim <> trainy.ndim or trainx.shape[1] <> trainy.shape[1]:
                    print 'the dimension of two object is not equal.'
                    return -1
                return float(numpy.dot(trainx[i].tolist()[0], trainy[j].tolist()[0]))
            
        except Exception as detail:
            print 'dot product error,detail:', detail
    
    @staticmethod
    def binary_search(collist,low,high,value):
        
        '''sorted list's binary search'''
        try:
            if low < 0 or high < 0 or low > high or len(collist) <= high or len(collist) <= low:
                return -1 
            
            if value < collist[low] or value > collist[high]:
                return -1
            if value == collist[low]:
                return low        
            if value == collist[high]:
                return high     
            l = low
            h = high
            while(l<=h):            
                mid = (l+h)/2
                if collist[mid] > value:
                    h = mid - 1
                elif collist[mid] < value:
                    l = mid + 1
                else:
                    return mid
        except Exception as detail:
            print 'binary_search error detail is:', detail
                    
        return -1
    
    @staticmethod 
    def convert(sparam, vec):
        
        '''To convert vector to matrix.'''
        
        if sparam.isdense == False:
            rows = [0]
            cols = []
            vals = []
            for i in range(len(vec)):
                if vec[i] <> 0:
                    cols.append(i)
                    vals.append(vec[i])
            rows.append(len(cols))
            return Matrix(rows, cols, vals, 1, len(vec))
        else:
            return matrix(vec)
        
    @staticmethod
    def RBF(sparam,trainx, trainy,xi,yi):
        '''the RBF kernel.'''
        
        paramlist = sparam.parameters
        eta = Svm_Util.dot(sparam, trainx, trainx,xi,xi)+Svm_Util.dot(sparam, trainy, trainy,yi,yi) - 2*Svm_Util.dot(sparam, trainx, trainy,xi,yi)
        res = 0.0
        if eta <0:
            res = math.exp(sparam.tolerance*float(paramlist[0]))
        else:
            res = math.exp(-eta*float(paramlist[0]))
        return res
        
    @staticmethod 
    def kernel_function(sparam, trainx, trainy):
        
        '''the kernel function.'''
        
        paramlist = sparam.parameters
        kernel_type = sparam.kernel_type
        
        if kernel_type == 'RBF':           
            return lambda xi,yi: Svm_Util.RBF(sparam,trainx, trainy,xi,yi)
        elif kernel_type == 'Linear':
            return lambda xi,yi:Svm_Util.dot(sparam, trainx, trainy,xi,yi) + float(paramlist[0])
        elif kernel_type == 'Polynomial':
            return lambda xi,yi: (float(paramlist[0]) * Svm_Util.dot(sparam, trainx, trainy, xi,yi) + float(paramlist[1])) ** int(paramlist[2])
        elif kernel_type == 'Sigmoid':
            return lambda xi,yi: math.tanh(float(paramlist[0]) * Svm_Util.dot(sparam, trainx, trainy,xi,yi) + float(paramlist[1]))
    
    @staticmethod
    def check_float_int(p,t):
        '''to Check the value of p can be transformed into a float (t = 0) or integer (t = 1).'''
        
        try:
            if t == 0:
                tmp = float(p) 
            elif t == 1:
                tmp = int(p)
        except:
            tmp = ''
            
        if (isinstance(tmp,float) and t == 0) or (isinstance(tmp,int) and t == 1):
            return True
        else:
            return False 
                
    @staticmethod
    def draw_scatter(xOffsets, yOffsets, xlabel = 'X', ylabel = 'Y', colors = None):   
        
        '''to draw draw_scatter picture.'''
        
        if (not isinstance(xOffsets,list)) or (not isinstance(yOffsets,list)):
            print 'xOffsets and yOffsets should be list type.'
            return 
        
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(0,1), ylim=(0,1)) 
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  
       
        if colors == None:
            ax.scatter(xOffsets, yOffsets)  
        else:
            ax.scatter(xOffsets, yOffsets, c=colors, alpha=0.75) 
        plt.show()
        file_name = 'mining/scatter_' + time.ctime() + '.png'
        plt.savefig(file_name)
        
    @staticmethod
    def draw_plot(xOffsets, yOffsets, xl = 'X', yl = 'Y', title = 'figure'):  
        
        '''to draw plot picture.''' 
        
        if (not isinstance(xOffsets,list)) or (not isinstance(yOffsets,list)):
            print 'xOffsets and yOffsets should be list type.'
            return 
        
        fig = plt.figure()   
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(0,1), ylim=(0,1))
        plt.xlabel(xl)
        plt.ylabel(yl) 
        ax.plot(xOffsets, yOffsets, lw=3, color='purple')
        plt.title(title)        
        plt.show()
        file_name = 'mining/plot_' + time.ctime() + '.png'
        plt.savefig(file_name)
               
class Smo_Csvc:
    '''a support vector machine classifier using 'C' to balance empirical risk and structural risk.'''
    
    def __init__(self,config, nodeName, loadFromFile = False, recoverFromLog = False, npRatio = 1):
        '''to initialize csvc.
        
            config: configuration file.
            nodeName: xml file's node.
            loadFromFile: Whether to read the csvc model from disk.
            recoverFromLog: Whether to recover the training procedure from disk.
            npRatio: negative samples / positive samples.
         '''        

        self.istrained = False
        #alpha
        self.alpha = []
        #gradient array
        self.G = []
        #weight
        self.w = []
        #bias
        self.b = 0.0
        #caching kii
        self.kcache = {}      
        #initialize svm model.
        self.model = Svm_Model(config, nodeName)  
        #negative samples number divide positive samples.
        self.npRatio = npRatio  
        
        #to get C1 and C2 for negative or positive samples.
        if self.npRatio > 1:
            self.C1 = self.model.config.C / self.npRatio
            self.C2 = self.model.config.C 
        else:
            self.C1 = self.model.config.C * self.npRatio
            self.C2 = self.model.config.C
        
        #to read model from disk.        
        if (loadFromFile):
            try:
                f = open(self.model.config.modelpath, "r")
                modelStr = pickle.load(f)
                self.model = pickle.loads(modelStr)
                f.close()
                self.istrained = True
            except IOError:
                pass
        
        #to recover training from log file.
        if recoverFromLog:    
            try:    
                f = open(self.model.config.logpath, "r")        
                modelStr = pickle.load(f)
                [self.alpha,self.G,self.w,self.b,self.model] = pickle.loads(modelStr)
                f.close()
            
            except IOError:
                pass
    
    def check_config(self):
    
        '''To check configuration file.'''
        
        kernel = ['Linear', 'RBF', 'Polynomial', 'Sigmoid']
        if self.model.config.kernel_type not in kernel:
            print '~kernel type error.'
            return False 
        
        if self.model.config.kernel_type == 'Linear' or self.model.config.kernel_type == 'RBF':
            if len(self.model.config.parameters) != 1:
                print '~Wrong number of parameters.'
                return False
            if not Svm_Util.check_float_int(self.model.config.parameters[0],0):
                print '~Parameter type error. detail:',self.model.config.parameters[0],'should be float type.'
                return False
            else:
                return True      
             
        if self.model.config.kernel_type == 'Polynomial':
            if len(self.model.config.parameters) != 3:
                print '~Wrong number of parameters.'
                return False
            if not (Svm_Util.check_float_int(self.model.config.parameters[0],0) and Svm_Util.check_float_int(self.model.config.parameters[1],0)):              
                 print '~Parameter type error. detail:',self.model.config.parameters[0], ' and ',self.model.config.parameters[1],'should be float type.'
                 return False
            elif not Svm_Util.check_float_int(self.model.config.parameters[2],1):
                 print '~Parameter type error. detail:',self.model.config.parameters[2], 'should be integer type.'
                 return False
            else:
                 return True
        
        if self.model.config.kernel_type == 'Sigmoid':
            if len(self.model.config.parameters) != 2:
                print '~Wrong number of parameters.'
                return False
            if not (Svm_Util.check_float_int(self.model.config.parameters[0],0) and Svm_Util.check_float_int(self.model.config.parameters[1],0)):
                 print '~Parameter type error. detail:',self.model.config.parameters[0], ' and ',self.model.config.parameters[1],'should be float type.'
                 return False
            else:
                return True 
            
    def GetValueFromKernelCache(self, i, i1, K, trainy):   
             
        '''To get kernel value from kernel cache.'''        
                         
        key1 = '%s%s%s'%(str(i1), '-', str(i))
        key2 = '%s%s%s'%(str(i),  '-', str(i1))

        if self.kcache.has_key(key1):
            k = self.kcache[key1]
        elif self.kcache.has_key(key2):
            k = self.kcache[key2]
        else:
            k = K(i1,i)
            if k < self.model.config.tolerance:
                k = 0             
            self.kcache[key1] = k  
            
        return k
    
    def ReduceCache(self):
        'To free memory & to prevent memory leaks.'
        
        try:
            newcache = {}
            if sys.getsizeof(self.kcache) > self.model.config.cachesize * (1024 **2):
                for key in self.kcache.iterkeys():
                    kl = key.split('-')
                    if kl[0] == kl[1]:
                          newcache[key] = self.kcache[key]
                self.kcache = 0
                self.kcache = newcache 
                print 'To free memory success.'
        except Exception as detail:
            print 'To free memory error,detail:', detail
         
    def SelectMaximumViolatingPair(self, trainy, K):
        
        '''To find the maximum violating pair from all samples.'''
                
        i = -1        
        G_max = float("-Infinity")
        obj_min = float("Infinity")
        
        for t in range(0, len(trainy)):
            if (trainy[t] == 1 and (self.C2 - self.alpha[t]) > self.model.config.tolerance ) or (trainy[t] == -1 and self.alpha[t] > 0):
                if -trainy[t] * self.G[t] >= G_max:
                    i = t
                    G_max =  -trainy[t] * self.G[t] 
        
        j = -1
        G_min = float("Infinity")
        for t in range(0, len(trainy)): 
            if (trainy[t] == -1 and (self.C1 - self.alpha[t]) > self.model.config.tolerance ) or (trainy[t] == 1 and self.alpha[t] > 0)  :
                b = G_max + trainy[t] * self.G[t]   
                     
                if -trainy[t] * self.G[t] <= G_min:                     
                    G_min = -trainy[t] * self.G[t]           
                if  b > 0:
                    a = 0.0
                    try:
                        a = self.GetValueFromKernelCache(i, i, K, trainy) +self.GetValueFromKernelCache(t, t, K, trainy) - 2 * self.GetValueFromKernelCache(i, t, K, trainy)
                        if a <= 0:
                            a = self.model.config.tolerance
                        if -(b*b)/(2*a) <= obj_min:  
                            j = t
                            obj_min = -(b*b)/(2*a)
                        
                    except Exception as detail:
                        print 'error detail is:', detail                   
        
        print 'Gap = ',G_max - G_min,'Fi=',trainy[i] * self.G[i],'Fj=',trainy[j] * self.G[j]  
            
        if G_max - G_min < self.model.config.eps:
            return [-1, -1, float("Infinity")]
        
        return [i, j, obj_min]        
     
    def W(self,trainy, alpha1new,alpha2newclipped,i,j,K):
    
        '''To calculate W value.'''

        alpha1 = self.alpha[i]
        alpha2 = self.alpha[j]
        y1 = trainy[i]
        y2 = trainy[j]
        s = y1 * y2
        
        k11 = self.GetValueFromKernelCache(i, i, K, trainy)
        k22 = self.GetValueFromKernelCache(j, j, K, trainy)
        k12 = self.GetValueFromKernelCache(i, j, K, trainy)
        
        w1 = alpha1new * (y1 * (-y1*self.G[i]) + alpha1 * k11 + s * alpha2 * k12)
        w1 += alpha2newclipped * (y2 * (-y2*self.G[j]) + alpha2 * k22 + s * alpha1 * k12)
        w1 = w1 - k11 * alpha1new * alpha1new/2 - k22 * alpha2newclipped * alpha2newclipped/2 - s * k12 * alpha1new * alpha2newclipped
        return w1   
    
    def calculate_auc(self,output,label):
        '''to calculate auc value.'''
        
        if output == None or label == None:
            return 0.0
        pos, neg = 0, 0         
        for i in range(len(label)):
            if label[i]>0:
                pos+=1
            else:    
                neg+=1
        output = sorted(output, key=itemgetter(0), reverse=True)
      
        tprlist = []
        fprlist = []
        tp, fp = 0., 0.            
        for i in range(len(output)):
            if output[i][1]>0:      
                tp+=1
            else:
                fp+=1
            tprlist.append(tp/pos)
            fprlist.append(fp/neg)
    
        auc = 0.            
        prev_rectangular_right_vertex = 0
        tpr_max = 0
        fpr_max = 0
        for i in range(0,len(fprlist)):
            if tpr_max < tprlist[i]:
                tpr_max = tprlist[i]
                
            if fpr_max < fprlist[i]:
                fpr_max = fprlist[i]
                
            if fprlist[i] != prev_rectangular_right_vertex:
                auc += (fprlist[i] - prev_rectangular_right_vertex) * tprlist[i]
                prev_rectangular_right_vertex = fprlist[i]         
        Svm_Util.draw_plot(fprlist, tprlist, 'FPR', 'TPR', 'ROC Curve(AUC = %.4f)' % auc)
        return auc
             
    def Train(self,trainx,trainy): 
               
        '''To train classifier.


            trainx is training matrix and trainy is classifying label'''      
        
        if self.model.config.isdense == False:
            if len(trainy) != trainx.nRow:
                print "ERROR!, trainx.nRow should == len(y)"
                return 0
        else:
            if trainx.shape[0] != len(trainy):
                print "ERROR!, trainx.shape[0] should == trainy.shape[0]"
                return 0
        
        #to check configuration.
        if not self.check_config():
            return [0,0]
            
        #to initialize all lagrange multipliers with zero.
        nrow = 0
        if self.model.config.isdense == True:
            nrow = trainx.shape[0]
        else:
            nrow = trainx.nRow
        ncol = 0
        if self.model.config.isdense == True:
            ncol = trainx.shape[1]
        else:
            ncol = trainx.nCol
                
        for i in range(0,nrow):
            self.alpha.append(0.0)
            
        for i in range(0,nrow):
            self.G.append(-1.0)
        #to initialize w with zero.
        for j in range(0,ncol):
            self.w.append(float(0))        
        
        #to get kernel function.
        K = Svm_Util.kernel_function(self.model.config, trainx, trainx)
        
        #the value of objective function.
        obj = 0.0
        #the iterations.
        iterations = 0        
        
        starttime = time.time()       
        while True:   
            begin = time.time()
            #to select maximum violating pair.
            [i, j, obj] = self.SelectMaximumViolatingPair(trainy, K) 
                
            if j == -1:
                break
        #-------------------------------------------------------------------begin to optimize lagrange multipiers i and j------------------------------------------------------- 
            L = 0.0 #the lower bound.
            H = 0.0 #the upper bound
            y1 = trainy[i]  #sample i's label.
            y2 = trainy[j]  #sample j's label.
            s = y1 * y2
            alpha1 = self.alpha[i] #sample i's alpha value.
            alpha2 = self.alpha[j] #sample j's alpha value.
            
            #to store old alpha value of sample i and j.
            oldalphai = self.alpha[i]   
            oldalphaj = self.alpha[j]    
            
            #the eta value.
            eta = self.GetValueFromKernelCache(i, i, K, trainy) +self.GetValueFromKernelCache(j, j,  K, trainy) - 2 * self.GetValueFromKernelCache(i, j, K, trainy)
                        
            #to calculate upper and lower bound.
            if y1*y2 == -1:
                gamma = alpha2 - alpha1
                if y1 == -1:
                    if gamma > 0:
                        L = gamma
                        H = self.C2
                    else:
                        L = 0
                        H = self.C1 + gamma   
                else:
                    if gamma > 0:
                        L = gamma
                        H = self.C1 
                    else:
                        L = 0
                        H = self.C2 + gamma

            if y1*y2 == 1:
                gamma = alpha2 + alpha1                
                if y1 == 1:
                    if gamma - self.C2 > 0:
                        L = gamma - self.C2
                        H = self.C2
                    else:
                        L = 0
                        H = gamma 
                else:
                    if gamma - self.C1 > 0:
                        L = gamma - self.C1
                        H = self.C1
                    else:
                        L = 0
                        H = gamma 
            
            if -eta < 0:
                #to calculate apha2's new value
                alpha2new = alpha2 + y2 * (y1*self.G[i] - y2*self.G[j])/eta
                
                if alpha2new < L:
                    alpha2newclipped = L
                elif alpha2new > H:
                    alpha2newclipped = H
                else:
                    alpha2newclipped = alpha2new
            else:            
                w1 = self.W(trainy, alpha1 + s * (alpha2 - L),L,i,j,K)
                w2 = self.W(trainy, alpha1 + s * (alpha2 - H),H,i,j,K)
                if w1 - w2 > self.model.config.eps:
                    alpha2newclipped = L
                elif w2 - w1 > self.model.config.eps:
                    alpha2newclipped = H
                else:
                    alpha2newclipped = alpha2              
             
            #to calculate aplha1
            alpha1new = alpha1 + s * (alpha2 - alpha2newclipped)
                                           
            if alpha1new < self.model.config.tolerance:
                alpha2newclipped += s * alpha1new
                alpha1new = 0
            elif y1 == -1 and alpha1new > self.C1:
                alpha2newclipped += s * (alpha1new - self.C1)
                alpha1new = self.C1
            elif y1 == 1 and alpha1new > self.C2:
                alpha2newclipped += s * (alpha1new - self.C2)
                alpha1new = self.C2
                   
            
                
            self.alpha[i] = alpha1new
            self.alpha[j] =  alpha2newclipped
            
            #to deal with Linear kernel.
            if self.model.config.kernel_type == 'Linear':
                ncol = 0
                if self.model.config.isdense == True:
                    ncol = trainx.shape[1]
                else:
                    ncol = trainx.nCol
                    
                if self.model.config.isdense == True:
                    self.w += (alpha1new - alpha1) * y1 * trainx[i] + (alpha2newclipped - alpha2) * y2 *trainx[j]
                else:
                    i1 = trainx.rows[i]
                    i2 = trainx.rows[j]
                    p1 = 0  #the elements number of row i
                    p2 = 0  #the elements number of row j
                    if i < len(trainx.rows)-1 :
                        p1 = trainx.rows[i+1] - trainx.rows[i]
                    
                    if j < len(trainx.rows)-1 :
                        p2 = trainx.rows[j+1] - trainx.rows[j]         
                        
                    for k in range(i1, i1+p1-1):
                        self.w[trainx.cols[k]] += (alpha1new - alpha1) * y1 * trainx.vals[k] 
                    for k in range(i2, i2+p2-1):
                        self.w[trainx.cols[k]] += (alpha2newclipped - alpha2) * y2 * trainx.vals[k]
            #-------------------------------------------------------------------end   to optimize lagrange multipiers i and j------------------------------------------------------- 
            deltaalphai = self.alpha[i] - oldalphai
            deltaalphaj = self.alpha[j] - oldalphaj           
            
            #to update gradient.
            for t in range(0, nrow):
                try:
                        part1 = trainy[t] * trainy[i] * self.GetValueFromKernelCache(t, i, K, trainy) * deltaalphai 
                        part2 = trainy[t] * trainy[j] * self.GetValueFromKernelCache(t, j, K, trainy) * deltaalphaj
                        self.G[t] += part1 + part2 
                except Exception as detail:
                    print 'error detail is:', detail
            
             
            print 'alpha', i, '=',self.alpha[i],'alpha', j,'=', self.alpha[j], 'the objective function value =', obj
           
            print time.time() - begin        
            iterations += 1    
            if iterations%self.model.config.times == 0:
                #dump to log file.
                f = open(self.model.config.logpath, "w")
                log = [self.alpha,self.G,self.w,self.b,self.model]
                modelStr = pickle.dumps(log,1)
                pickle.dump(modelStr, f)
                f.close() 
                self.ReduceCache()
               
        #To store support vectors.
        index = []
        for i in range(0, len(self.alpha)):
            if self.alpha[i] > 0:
                index.append(i)
                self.model.alpha.append(self.alpha[i])
                self.model.label.append(trainy[i])
        self.model.svn = len(index)
        self.model.w = self.w
        
        #--------------------------------------------------------
        b1 = 0.0
        b2 = 0.0
        c1 = 0
        for i in range(0,len(index) ):
            if trainy[index[i]] == -1:
                b1 += -trainy[index[i]] * self.G[index[i]]
                c1 += 1
            else:
                b2 += -trainy[index[i]] * self.G[index[i]]
        
        self.b = ((b1/c1)+(b2/(self.model.svn - c1)))/2
        self.model.b = self.b      
        print 'the threshold value =', self.b
        #--------------------------------------------------------
        
        #to store support vector machines.
        if self.model.config.isdense == True:
            sv = []                
            for i in range(0, len(index)):
                sv.append(trainx[index[i]].tolist()[0])
            self.model.sv = matrix(sv)
            
        else:
            rows = []
            cols = []
            vals = []
            pos = 0
            for i in range(0, len(index)):
                i1 = trainx.rows[index[i]]
                p1 = 0
                
                if index[i] < len(trainx.rows)-1 :
                    p1 = trainx.rows[index[i]+1] - trainx.rows[index[i]]
                
                k = 0
                while(k < p1):                    
                    cols.append(trainx.cols[i1 + k])
                    vals.append(trainx.vals[i1 + k])
                    k += 1
                rows.append(pos)
                pos += p1     
            rows.append(len(vals))
            self.model.sv = Matrix(rows, cols, vals ,self.model.svn, trainx.nCol )

        #dump model path
        f = open(self.model.config.modelpath, "w")
        modelStr = pickle.dumps(self.model, 1)
        pickle.dump(modelStr, f)
        f.close()            
        self.istrained = True
        
        try:
            os.remove(self.model.config.logpath)
        except:
            pass           
                               
        return [time.time()-starttime,iterations]  
        
    def Test(self,testx,testy):
        
        '''To test samples.


            self.testx is training matrix and self.testy is classifying label'''    

        TP = 0.0
        TN = 0.0
        FP = 0.0
        FN = 0.0
        Recall = 0.0
        Precision = 0.0
        Accuracy = 0.0
        Fbeta1 = 0.0
        Fbeta2 = 0.0
        AUCb = 0.0
        
        TPR= 0.0
        FPR = 0.0
        pn = 0.0
        nn = 0.0
        tprlist = []
        fprlist = []
        
        outputlist = []
        
        for i in range(len(testy)):
            if testy[i] == 1:
                pn = pn + 1
            else:
                nn = nn + 1
        
        #check parameter
        if (not self.istrained):
            print "Error!, not trained!"
            return False 
   
        K = Svm_Util.kernel_function(self.model.config, self.model.sv, testx)       
        nrow = 0
        if self.model.config.isdense == True:
            nrow = testx.shape[0]
        else:
            nrow = testx.nRow
            
        for i in range(0, nrow):
            fxi = 0.0
            if self.model.config.kernel_type == 'Linear':
                fxi = Svm_Util.dot(self.model.config, Svm_Util.convert(self.model.config, self.model.w), testx, 0, i) + self.model.b
            else:
                for j in range(0, self.model.svn):
                    fxi += self.model.alpha[j] * self.model.label[j] * K(j, i) 
                fxi += self.model.b
            
            if testy[i] == 1 and fxi >=0:
                TP += 1
            if testy[i] == -1 and fxi <=0:
                TN += 1
            if testy[i] == -1 and fxi >=0:
                FP += 1
            if testy[i] == 1 and fxi <=0:
                FN += 1                
            
            #to calculate ROC value.  
            TPR = TP/pn
            FPR = FP/nn
            tprlist.append(TPR)  
            fprlist.append(FPR)        
            
            outputlist.append([fxi,testy[i]])
            print i,': Actual output is', fxi, 'It\'s label is', testy[i]
        
        #to calculate auc
        auc = 0.            
        prev_rectangular_right_vertex = 0
        tpr_max = 0
        fpr_max = 0
        for i in range(0,len(fprlist)):
            if tpr_max < tprlist[i]:
                tpr_max = tprlist[i]
                
            if fpr_max < fprlist[i]:
                fpr_max = fprlist[i]
                
            if fprlist[i] != prev_rectangular_right_vertex:
                auc += (fprlist[i] - prev_rectangular_right_vertex) * tprlist[i]
                prev_rectangular_right_vertex = fprlist[i] 
        
        try:        
            Recall = TP/(TP + FN)
            Precision = TP/(TP + FP)
            Accuracy = (TP + TN)/(TP + TN + FP + FN)
            Fbeta1 = 2 * (Recall * Precision)/(1 + Precision + Recall)
            Fbeta2 = 5 * (Recall * Precision)/(4 + Precision + Recall)
            AUCb = (Recall + TN/(FP + TN))/2
            
            print 'Recall = ', Recall, 'Precision = ', Precision,'Accuracy = ', Accuracy,'\n', 'F(beta=1) = ', Fbeta1, 'F(beta=2) = ', Fbeta2, 'AUCb = ',AUCb
            
        except Exception as detail:
            print 'to test error,detail is:', detail
            
        self.calculate_auc(outputlist,testy)
            
        return [Recall,Precision,Accuracy,Fbeta1,Fbeta2,AUCb,auc]
