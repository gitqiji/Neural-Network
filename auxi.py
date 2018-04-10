import numpy as np
import cv2

class Auxi:
    def data0(self):
        col = 100;row = 2
        x = np.random.randint(10,300,size = (row,col))
        t = x[0] - x[1]
        y = np.where(t > 0,0,1).reshape(1,col)
        return x,y
    def data1(self):
        row = 2
        y = np.zeros((1,100))
        y[0,0:50] = 1
        y[0,50:] = 0
        x0 = np.random.randint(10,150,size=(row,50))
        t0 = np.random.randint(150,300,size=(row,1,15))
        t1 = np.random.randint(10,150,size=(row,1,15))
        xtR = np.vstack ((t0[0],t1[1]))#np.append(t0[0],t1[1],axis = 0)
        xtL = np.vstack ((t1[1],t0[0]))#np.append(t0[1],t1[0],axis = 0)
        xt1 = np.hstack((xtR,xtL))#np.append(xtR,xtL,axis = 1)
        xt0 = np.random.randint(150,300,size=(row,20))
        x1 = np.hstack ((xt0,xt1))#np.append (xt0,xt1,axis = 1)
        x = np.append (x0,x1,axis = 1)
        return x,y
    
    def plotinitial(self,flag,x):
        m = x.shape[1]
        img = np.zeros((500,500,3),'uint8')
        if flag == 0:
            img = cv2.line(img,(0,0),(500,500),(255,255,255),1)
                
        else:
            img = cv2.line(img,(150,0),(150,150),(255,255,255),1)
            img = cv2.line(img,(0,150),(150,150),(255,255,255),1)
        
        return img,m
    def plottrain(self,flag,x,A):
        img,m = self.plotinitial(flag,x)
        idx = len(A) - 1
        for i in range(m):
            img = cv2.circle(img,(x[:,i][0],x[:,i][1]),5,(255,255,255),0)
            if round(A[idx][0][i]) == 0:
                img = cv2.circle(img,(x[:,i][0],x[:,i][1]),4,(0,0,255),-1)
            else:
                img = cv2.circle(img,(x[:,i][0],x[:,i][1]), 4, (0,255,0), -1)
                
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'label: 0',(10,450), font, 1,(255,255,255),1,cv2.LINE_AA)
        img = cv2.rectangle(img,(150,430),(160,450), (0,0,255), -1,8,0)
    
        cv2.putText(img,'label: 1',(300,30), font, 1,(255,255,255),1,cv2.LINE_AA)
        img = cv2.rectangle(img,(440,10),(450,30), (0,255,0), -1,8,0)
        
        cv2.imshow('DnnNet',img)
        cv2.waitKey (0)
        cv2.destroyAllWindows()
