import numpy as np
import cv2
def sigmod(Z,flag):
    if flag == 0:
        return 1/(1+np.exp(-Z))
    else:
        return Z * (1 - Z)
def relu(Z,flag):
    if flag == 0:
        return np.maximum(0,Z)
    else:
        return np.where(Z > 0,1,0)
def inputz(X,W,B):    
    Z = np.dot(W,X) + B
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
def initialwb():
    W2 = np.random.randn(1,4) * 0.01
    B2 = np.random.randn(1,1)
    W1 = np.random.randn(4,2) * 0.01
    B1 = np.random.randn(4,1)
    
    return W2,B2,W1,B1

def forward(X,Y,W2,B2,W1,B1,af):
    Z1 = inputz(X,W1,B1)#4xm
    if af == 0:
        A1 = sigmod(Z1,0)#4xm
    else:
        A1 = relu(Z1,0)#4xm
    Z2 = inputz(A1,W2,B2)#1xm
    A2 = sigmod(Z2,0)#1xm
    los = lossf(Y,A2)
    
    return A1,A2,los

def train(m,X,Y,W2,B2,W1,B1,af):
    A1,A2,los = forward(X,Y,W2,B2,W1,B1,af)
    dZ2 = A2 - Y #1xm
    dW2 = np.dot(dZ2,A1.T)/m #1x4
    dB2 = dZ2.sum(axis = 1,keepdims = True)/m
    
    dA1 = np.dot(W2.T,dZ2) #4xm
    if af == 0:
        dZ1 = dA1 * sigmod(A1,1) #4xm
    else:
        dZ1 = dA1 * relu(A1,1) #4xm
    dW1 = np.dot(dZ1,X.T)/m #4x2
    dB1 = dZ1.sum(axis = 1,keepdims = True)/m

    res = [round(em) for em in A2[0]] - Y[0]
    print res
    return dW2,dB2,dW1,dB1,A2,los/m

def updatewb(W2,B2,W1,B1,dW2,dB2,dW1,dB1):
    alpha = 0.2
    NW2 = W2 - alpha * dW2
    NB2 = B2 - alpha * dB2
    NW1 = W1 - alpha * dW1
    NB1 = B1 - alpha * dB1
    
    return NW2,NB2,NW1,NB1
def producedata():
    y = np.zeros((1,100))
    while y.__contains__(0):
        x = np.random.randint(10,300,size = (2,100))
        y = x[0] - x[1]
    
    y = np.where(y > 0,1,0).reshape(1,100)
    return x,y

def process():
    W02,B02,W01,B01 = initialwb()
    X,Y = producedata()
    m = X.shape[1]
    
    W2=W02; B2=B02; W1=W01; B1=B01
    for itr in range(500):
        
        dW2, dB2, dW1, dB1, A2, los = train(m,X/10,Y,W2,B2,W1,B1,0)
            
        print 'sigmod iter %d: loss is: %f' % (itr,los/m)
        if itr % 10 == 0:
            plot(X,Y,W2,B2,W1,B1,A2,1)
        W2,B2,W1,B1 = updatewb(W2,B2,W1,B1,dW2,dB2,dW1,dB1)
      
    W2=W02; B2=B02; W1=W01; B1=B01
    for itr in range(500):
        
        dW2, dB2, dW1, dB1, A2, los = train(m,X/10,Y,W2,B2,W1,B1,1)
            
        print 'RELU iter %d: loss is: %f' % (itr,los/m)
        if itr % 10 == 0:
            plot(X,Y,W2,B2,W1,B1,A2,1)
        W2,B2,W1,B1 = updatewb(W2,B2,W1,B1,dW2,dB2,dW1,dB1)
        
def plot(X,Y,W2,B2,W1,B1,A2,flag):
    img = np.zeros((500,500,3), np.uint8)
    img = cv2.line(img,(0,0),(500,500),(255,255,255),1,4-4)
    for i in range(100):
        #initialize
        img = cv2.circle(img,(X[0,i],X[1,i]), 5, (255,255,255), 0)
        if flag == 1:
            if round(A2[0][i]) == 0:
                img = cv2.circle(img,(X[0,i],X[1,i]), 4, (0,0,255), -1)
            else:
                img = cv2.circle(img,(X[0,i],X[1,i]), 4, (0,255,0), -1)
       
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'label: 0',(10,450), font, 1,(255,255,255),1,cv2.LINE_AA)
    img = cv2.rectangle(img,(150,430),(160,450), (0,0,255), -1,8,0)
    
    cv2.putText(img,'label: 1',(300,30), font, 1,(255,255,255),1,cv2.LINE_AA)
    img = cv2.rectangle(img,(440,10),(450,30), (0,255,0), -1,8,0)
    
    cv2.imshow('nnvector',img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    
    process()
        
