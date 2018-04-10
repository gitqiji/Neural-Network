import numpy as np
class net:
    def __init__(self,layer,node):
        self.lynum  = layer
        lyndnum = []
        lyndnum.append(2) #input
        W = {};B = {}
        for i in range(self.lynum):
            L = i + 1
            lyndnum.append(node[i])
            W[L] = np.random.randn(lyndnum[L],lyndnum[L-1])
            B[L] = np.random.randn(lyndnum[L],1)
        self.W = W
        self.B = B

    def sigmod(self,Z,flag):
        if flag == 0:
            return 1/(1+np.exp(-Z))
        else:
            return Z * (1 - Z)
    def lossf(self,Y,A):
        T = Y + A
        if T.__contains__(1):
            los = float('inf')
        elif A.__contains__(0) or A.__contains__(1):
            los = 0
        else:
            los = -(np.dot(Y,np.log(A.T)) + np.dot(1-Y,np.log(1-A.T)))
        return los
    def forward(self,X,Y,W,B):
        A = {};Z = {}
        #4 4 1 forward
        A[0] = X
        for i in range(self.lynum):
            L = i + 1
            #print 'fL:',L
            Z[L] = np.dot(W[L],A[L-1]) + B[L]
            A[L] = self.sigmod(Z[L],0)
        
        los = self.lossf(Y,A[L])
        return A,los
    def train(self,m,X,Y,W,B):
        A,los = self.forward(X,Y,W,B)
        L = self.lynum; nL = L
        
        dA = {};dZ = {};dW = {};dB = {}
        while L > 0:
            print 'bL:',L
            if L == nL:
                dZ[L] = A[L] - Y#output layer
            else:
                dA[L] = np.dot(W[L+1].T,dZ[L+1])#ndnum x m
                dZ[L] = dA[L] * self.sigmod(A[L],1)#ndnum x m
            dW[L] = np.dot(dZ[L],A[L-1].T)/m
            dB[L] = dZ[L].sum(axis = 1,keepdims = True)/m
            L = L - 1

        res = [round(em) for em in A[nL][0]] - Y[0]
        print res
    
        return dW,dB,los/m,A
    def updatewb(self,L,W,B,dW,dB,itr):
        if itr < 500:
            alpha = 0.2
        else:
            alpha = 0.02
        NW = {};NB = {}
        for i in range(L):
            t = i + 1
            NW[t] = W[t] - alpha * dW[t]
            NB[t] = B[t] - alpha * dB[t]
    
        return NW,NB
    def process(self,X,Y,itr):
        W = self.W
        B = self.B
        
        m = X.shape[1]
        
        dW, dB, los, A = self.train(m,X/10,Y,W,B)
            
        self.W,self.B = self.updatewb(self.lynum,W,B,dW,dB,itr)
        return A,los
