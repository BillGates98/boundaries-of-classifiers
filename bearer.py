import numpy as np
from scipy.special import factorial

class Wave:
    def __init__(self, x=[]):
        self.x = x
    
    def sigmoid(self, z):
        return 1 / ( 1 + np.exp(-z))

    def c(self):
        output = []
        for i in range(len(self.x)) :
            t = self.x[i]
            value = np.cos(t*np.pi+1) # / ( 1 + np.exp(t))
            output.append(value)
        return output

    def s(self):
        output = []
        for i in range(len(self.x)) :
            t = self.x[i]
            value = np.cos(t*np.pi+1) #/ ( 1 + np.exp(t))
            output.append(value)
        return output

    def error(self, weight=[], h=1):
        c = self.c()
        output = [ c[i] for i in range(len(c))]
        prediction = self.sigmoid(np.dot(output, weight))
        output = abs(h-prediction)
        return output
    
    def _error(self, weight=[], h=1):
        s = self.c()
        output = [ s[i] for i in range(len(s))]
        prediction = self.sigmoid(np.dot(output, weight))
        output = abs(h-prediction)
        return output

    def predict(self, weight=[], h=1, me=0.0, std=0.0, y=0):
        output = 0
        error = self._error(weight=weight, h=h)
        # gap = (std/error)*100
        # print('X = ', self.x)
        # print('\t \t Pearson : ', (std/error)*100 )
        # print('\t Pearson : ', (std/m_error)*100,  'Error : ', round(error,2), ' and Me : ', round(m_error,2), ' Gap : ', round(gap,2), ' ', ' std : ', round(std,2), ' Truth : ', y)
        # if gap >= 0  :
        #     output = 1
        # else:
        #     output = 0
        return error # output
        
    
