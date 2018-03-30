import numpy as np
import cv2
def sigmod(x):
    return 1/(1+np.exp(-x))
def inputz(x,w,b):
    t = w * x
    z = t.sum() + b
    return z
def lossf(y,a):
    if a == 0 or a == 1:
        los = 0
    else:
        los = -(y*np.log(a) + (1-y)*np.log(1-a))
    return los
def train(x,y,w,b):
    z = inputz(x,w,b)
    a = sigmod(z)
    los = lossf(y,a)
    dz = a - y
    dw = dz * x
    db = dz
    return dw, db, los
def updatewb(w0,b0,avw,avb):
    alpha = 0.2
    w1 = w0 - alpha * avw
    b1 = b0 - alpha * avb
    return w1,b1
def process():
    w = np.random.randn(1,2)
    b = np.random.randn(1,1)
    sampleset = np.array([[10,20],[20,50],[30,60],[30,40],[80,120],
                          [130,100],[20,10],[30,20],[40,10]])
    labelset = np.array([[1],[1],[1],[1],[1],[0],[0],[0],[0]])
    sumdw = 0; sumdb = 0; sumlos = 0;
    m = sampleset.shape[0]
    plot(sampleset,labelset,w,b,0)

    for itr in range(20):
        for x in zip(sampleset,labelset):
            dw, db, los = train(x[0],x[1],w,b)
            sumdw += dw
            sumdb += db
            sumlos += los
        print 'iter %d: loss is: %f' % (itr,sumlos/m)
        w, b = updatewb(w,b,sumdw/m,sumdb/m)
        sumdw = 0; sumdb = 0; sumlos = 0;
        plot(sampleset,labelset,w,b,1)
        
def plot(sampleset,labelset,w,b,flag):
    img = np.zeros((500,500,3), np.uint8)
    img = cv2.line(img,(0,0),(500,500),(255,255,255),1,0)
    for sm in zip(sampleset,labelset):
        if sm[1] == 1:
            img = cv2.circle(img,(sm[0][0],sm[0][1]), 5, (255,255,255), 0)
        else:
            img = cv2.circle(img,(sm[0][0],sm[0][1]), 5, (255,255,255), 0)
    if flag == 1:      
        for sm in zip(sampleset,labelset):
            flag = inputz(sm[0],w,b)
            if flag > 0.5:
                img = cv2.circle(img,(sm[0][0],sm[0][1]), 4, (0,0,255), -1)
            else:
                img = cv2.circle(img,(sm[0][0],sm[0][1]), 4, (0,255,0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'label is : 0',(10,450), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img,'label is : 1',(300,30), font, 1,(255,255,255),2,cv2.LINE_AA)
    
    cv2.imshow('logicRegression',img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    process()
