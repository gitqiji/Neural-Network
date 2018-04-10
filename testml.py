from auxi import Auxi
from dnnnet import net

if __name__ == '__main__':
    auxis = Auxi()
    X,Y = auxis.data1()
    print X.shape,Y.shape
 
    Net = net(3,(5,4,1))
    for itr in range(1000):
        A,los = Net.process(X,Y,itr)
        print 'iter %d: loss is: %f' % (itr,los)
        if itr % 10 == 0:
            auxis.plottrain(1,X,A)
