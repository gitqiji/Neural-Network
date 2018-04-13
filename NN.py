import numpy as np
import cv2
result = []
def sigmod(Z):
    return 1/(1+np.exp(-Z))

def inputz(X,W,B):
    
    Z = np.dot(W,X) + B
    return Z
def lossf(y,a):
    global result
    if a <= 0.00001 and y == 0 or a >= 0.99999 and y == 1:
        los = 0
    elif a == 0 or a == 1:
        los = float('inf')
    else:
        #los = -(np.dot(y,np.log(a)) + np.dot(1-y,np.log(1-a)))
        los = -(y*np.log(a) + (1-y)*np.log(1-a))
    result.append(round(a))
    return los

def train(X,Y,W2,B2,W1,B1):
    Z1 = inputz(X,W1,B1)#4x1
    #print 'Z1.shape',Z1.shape
    A1 = sigmod(Z1)#4x1
    #print 'A1.shape',A1.shape
    Z2 = inputz(A1,W2,B2)#1x1
    #print 'Z2.shape',Z2.shape
    A2 = sigmod(Z2)#1x1
    #print 'A2.shape',A2.shape
    
    dZ2 = A2 - Y #1x1
    dW2 = np.dot(dZ2,A1.T) #1x4
    #print 'dW2.shape',dW2.shape
    dB2 = dZ2
    
    dA1 = np.dot(W2.T,dZ2) #4x1
    dZ1 = dA1 * A1 * (1 - A1) #4x1
    dW1 = np.dot(dZ1,X.T) #4x2
    dB1 = dZ1

    los = lossf(Y,A2)

    return dW2,dB2,dW1,dB1,los

def updatewb(W2,B2,W1,B1,dW2,dB2,dW1,dB1):
    alpha = 0.2
    NW2 = W2 - alpha * dW2
    NB2 = B2 - alpha * dB2
    NW1 = W1 - alpha * dW1
    NB1 = B1 - alpha * dB1
    
    return NW2,NB2,NW1,NB1
def test(X,W2,B2,W1,B1):
    
    Z1 = inputz(X,W1,B1)#4x1
    A1 = sigmod(Z1)#4x1
    Z2 = inputz(A1,W2,B2)#1x1
    A2 = sigmod(Z2)#1x1
    return A2
def process():
    W2 = np.random.randn(1,4)
    B2 = np.random.randn(1,1)
    W1 = np.random.randn(4,2)
    B1 = np.random.randn(4,1)
    sampleset = np.array([[10,20],[20,50],[30,60],[30,40],[80,120],
                          [130,100],[20,10],[30,20],[40,10]])
    labelset = np.array([[1],[1],[1],[1],[1],[0],[0],[0],[0]])
    global result
    m = sampleset.shape[1]
    for itr in range(30):
        sumdW2 = np.zeros((1,4));sumdB2 = np.zeros((1,1))
        sumdW1 = np.zeros((4,2));sumdB1 = np.zeros((4,1))
        sumlos = 0
        for x in zip(sampleset,labelset):
            x0 = x[0].reshape(1,2)
            #print 'x.shape',x0.shape,type(x0)
            dW2, dB2, dW1, dB1, los = train(x0.T,x[1],W2,B2,W1,B1)
            sumdW2 += dW2
            sumdB2 += dB2
            sumdW1 += dW1
            sumdB1 += dB1
            
            sumlos += los
        print 'iter %d: loss is: %f' % (itr,sumlos/m)
        print result
        result = []
        W2,B2,W1,B1 = updatewb(W2,B2,W1,B1,sumdW2/m,sumdB2/m,sumdW1/m,sumdB1/m)

if __name__ == '__main__':
    
    process()
        
