import numpy as np

def sigmod(Z):
    return 1/(1+np.exp(-Z))
def tanh(Z):
    return (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
def inputz(X,W,B):    
    Z = np.dot(W,X) + B
    return Z
def lossf(Y,A):
    T = Y + A
    if T.__contains__(1):
        los = float('inf')
    elif A.__contains__(0) or A.__contains__(1):
        los = 0
    else:
        los = -(np.dot(Y,np.log(A.T)) + np.dot(1-Y,np.log(1-A.T)))
    return los
def train(m,X,Y,W2,B2,W1,B1):
    Z1 = inputz(X,W1,B1)#4xm
    #print 'Z1.shape',Z1.shape
    A1 = tanh(Z1)#4xm
    #print 'A1.shape',A1.shape
    Z2 = inputz(A1,W2,B2)#1xm
    #print 'Z2.shape',Z2.shape
    A2 = sigmod(Z2)#1xm
    #print 'A2.shape',A2.shape
    
    dZ2 = A2 - Y #1xm
    dW2 = np.dot(dZ2,A1.T)/m #1x4
    #print 'dW2.shape',dW2.shape
    dB2 = dZ2.sum(axis = 1,keepdims = True)/m
    
    dA1 = np.dot(W2.T,dZ2) #4xm
    dZ1 = dA1 * (1 - A1) * (1 - A1) #4xm
    dW1 = np.dot(dZ1,X.T)/m #4x2
    dB1 = dZ1.sum(axis = 1,keepdims = True)/m

    los = lossf(Y,A2)
    '''
    print 'ground  truth',[round(em) for em in Y[0]]
    print 'predict value',[round(em) for em in A2[0]]
    '''
    res = [round(em) for em in A2[0]] - Y[0]
    print res
    
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
def producedata():
    y = np.zeros((1,100))
    while y.__contains__(0):
        x = np.random.randint(10,300,size = (2,100))
        y = x[0] - x[1]
    
    y = np.where(y > 0,0,1).reshape(1,100)
    return x,y
def process():
    W2 = np.random.randn(1,4)
    B2 = np.random.randn(1,1)
    W1 = np.random.randn(4,2)
    B1 = np.random.randn(4,1)
    X,Y = producedata()
    X = X / 10
    m = X.shape[1]
    for itr in range(20):
        
        dW2, dB2, dW1, dB1, los = train(m,X,Y,W2,B2,W1,B1)
            
        print 'iter %d: loss is: %f' % (itr,los/m)
        
        W2,B2,W1,B1 = updatewb(W2,B2,W1,B1,dW2,dB2,dW1,dB1)

if __name__ == '__main__':
    
    process()
