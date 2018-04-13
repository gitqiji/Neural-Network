import numpy as np
import cv2
def sigmod(Z):
    return 1/(1+np.exp(-Z))

def inputz(X,W,B):
    
    Z = np.dot(W.T,X) + B#1xm
    return Z
def lossf(Y,A):
    T = Y + A
    if T.__contains__(1):
        los = float('inf')
    elif T.__contains__(0) or T.__contains__(2):
        los = 0
    else:
        los = -(np.dot(Y,np.log(A.T)) + np.dot(1-Y,np.log(1-A.T)))
    return los
def forward(X,Y,W,B):
    Z = inputz(X,W,B)#1xm
    A = sigmod(Z)#1xm
    los = lossf(Y,A)
    return A,los
def train(m,X,Y,W,B):
    A,los = forward(X,Y,W,B)
    
    dZ = A - Y #1xm'''
    dW = np.dot(X,dZ.T)/m#2x1
    dB = np.sum(dZ,axis = 1,keepdims = True)/m#1x1
   
    res = [round(em) for em in A[0]] - Y[0]
    print res
    
    return dW,dB,los/m
def updatewb(W0,B0,dW,dB):
    alpha = 0.2
    W1 = W0 - alpha * dW
    B1 = B0 - alpha * dB
    return W1,B1

def producedata():
    y = np.zeros((1,100))
    while y.__contains__(0):
        x = np.random.randint(10,300,size = (2,100))
        y = x[0] - x[1]
    
    y = np.where(y > 0,1,0).reshape(1,100)
    return x,y
def process():
    W = np.random.randn(2,1)
    B = np.random.randn(1,1)
    X,Y = producedata()
    
    m = X.shape[1]
    for itr in range(100):
        dW,dB,los = train(m,X/10,Y,W,B)
        W,B = updatewb(W,B,dW,dB)
        print 'loss in iter %d : %f'%(itr,los)
        plot(X,Y,W,B,1)
def plot(X,Y,w,b,flag):
    img = np.zeros((500,500,3), np.uint8)
    img = cv2.line(img,(0,0),(500,500),(255,255,255),1,4-4)
    for i in range(100):
        #initialize
        if Y[0][i] == 1:
            img = cv2.circle(img,(X[0,i],X[1,i]), 5, (255,255,255), 0)
        else:
            img = cv2.circle(img,(X[0,i],X[1,i]), 5, (255,255,255), 0)
        if flag == 1:
            A,los = forward(X[:,i].reshape(2,1)/10,Y[:,i].reshape(1,1),w,b)
            if A[0][0] < 0.5:
                img = cv2.circle(img,(X[0,i],X[1,i]), 4, (0,0,255), -1)
            else:
                img = cv2.circle(img,(X[0,i],X[1,i]), 4, (0,255,0), -1)
        
   
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'label: 0',(10,450), font, 1,(255,255,255),1,cv2.LINE_AA)
    img = cv2.rectangle(img,(150,430),(160,450), (0,0,255), -1,8,0)
    
    cv2.putText(img,'label: 1',(300,30), font, 1,(255,255,255),1,cv2.LINE_AA)
    img = cv2.rectangle(img,(440,10),(450,30), (0,255,0), -1,8,0)
    
    cv2.imshow('logicVector',img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    process()
        
