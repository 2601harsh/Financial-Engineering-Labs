import random
import math
import numpy as np
import matplotlib.pyplot as plt

M = np.array([0.1, 0.2, 0.15])
C = np.array([[0.005, -0.010, 0.004],[-0.010, 0.040, -0.002],[0.004, -0.002, 0.023]])
CI = np.linalg.inv(C)
U = np.array([1, 1, 1])
def getW(mu):
    return (np.linalg.det([[1, U @ CI @ M.T],[mu, M @ CI @ M.T]])*(U @ CI) + np.linalg.det([[U @ CI @ U.T, 1],[M @ CI @ U.T, mu]])*(M @ CI))/(np.linalg.det([[U @ CI @ U.T, U @ CI @ M.T],[M @ CI @ U.T, M @ CI @ M.T]]))
def getSig(W):
    return math.sqrt(W @ C @ W.T)
def getMinVar():
    return getSig((U @ CI)/(U @ CI @ U.T))
# a
X = []
Y = []
for i in np.linspace(0,getMinVar()+0.5,5000):
    X.append(getSig(getW(i)))
    Y.append(i)
plt.plot(X,Y)
plt.xlabel("Risk (sigma)")
plt.ylabel("Returns") 
plt.title("Minimum variance line along with Markowitz Efficient Frontier")
plt.legend()
plt.show()
# b
print("Return                 Standard Deviation         W1          W2           W3")
for i in np.linspace(getMinVar()-0.1,getMinVar()+0.5,10):
    W = getW(i)
    print(i,getSig(W),W[0],W[1],W[2])
# c
difmi = 100
valmi = 0
difma = 100
valma = 0
for i in range(len(X)):
    if(Y[i] < 0.06):
        if(difmi > abs(X[i]-0.15)):
            difmi = abs(X[i]-0.15)
            valmi = Y[i]
    if(Y[i] > 0.15):
        if(difma > abs(X[i]-0.15)):
            difma = abs(X[i]-0.15)
            valma = Y[i]
print("for 15 percent risk")
print("minimum return  = ", valmi)
print("W = ",getW(valmi)[0],getW(valmi)[1],getW(valmi)[2])
print("maximum return  = ", valma)
print("W = ",getW(valma)[0],getW(valma)[1],getW(valma)[2])
#d
print("For a 18 percent return, the minimum risk portfolio is")
print("minimum risk  = ", getSig(getW(0.18))*100)
print("W = ",getW(0.18)[0],getW(0.18)[1],getW(0.18)[2])
#e
max_ri = 0
max_re = 0
sr = -1e9
for i in range(len(X)):
    if (Y[i]-0.1)/X[i] > sr:
        sr = (Y[i]-0.1)/X[i]
        max_re = Y[i]
        max_ri = X[i]
print("Market portfolio has risk = ", max_ri)
print("Market portfolio has return = ", max_re)
print("W = ",getW(max_re)[0],getW(max_re)[1],getW(max_re)[2])

nX = []
nY = []
for i in np.linspace(0,0.5,1000):
    nX.append(i)
    nY.append(0.1+sr*i)
plt.plot(nX,nY)
plt.plot(X,Y)
plt.xlabel("Risk (sigma)")
plt.ylabel("Returns") 
plt.title("Capital Market Line with Minimum variance curve")
plt.grid(True)
plt.show()
# f
