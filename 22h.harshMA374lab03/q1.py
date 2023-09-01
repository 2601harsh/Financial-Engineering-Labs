import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def bm(S0, K, T, r, si, M):
    dt=T/M
    sqrtdt = math.sqrt(dt)
    u = np.exp(si * np.sqrt(dt) + (r - (si * si) / 2) * dt)
    d = np.exp(-si * np.sqrt(dt) + (r - (si * si) / 2) * dt)
    R = math.e**(r*dt)
    p = (R-d)/(u-d)
    q = 1-p
    
    val = np.zeros((M+1,M+2))
    val[0][0]=S0
    for i in range(0,M):
        for j in range(0,(i+1)):
            val[i+1][j]=u*val[i][j]
        val[i+1][i+1]=val[i][i]*d
    
    pay = np.zeros((M+1,M+2))

    for i in range(M+1):
        pay[M][i]=max(0,val[M][i]-K)

    d = np.exp(-r*dt)
    for i in range(M-1,-1,-1):
        for j in range(i+1):
            pay[i][j]=max(val[i][j]-K,d*(p*pay[i+1][j]+q*pay[i+1][j+1]))
    return pay[0][0]

def bm2(S0, K, T, r, si, M):
    dt=T/M
    sqrtdt = math.sqrt(dt)
    u = np.exp(si * np.sqrt(dt) + (r - (si * si) / 2) * dt)
    d = np.exp(-si * np.sqrt(dt) + (r - (si * si) / 2) * dt)
    R = math.e**(r*dt)
    p = (R-d)/(u-d)
    q = 1-p
    
    val = np.zeros((M+1,M+2))
    val[0][0]=S0
    for i in range(0,M):
        for j in range(0,(i+1)):
            val[i+1][j]=u*val[i][j]
        val[i+1][i+1]=val[i][i]*d
    
    pay = np.zeros((M+1,M+2))

    for i in range(M+1):
        pay[M][i]=max(0,K-val[M][i])

    d = np.exp(-r*dt)
    for i in range(M-1,-1,-1):
        for j in range(i+1):
            pay[i][j]=max(K-val[i][j],d*(p*pay[i+1][j]+q*pay[i+1][j+1]))
    return pay[0][0]

print(bm(100,100,1,0.08,0.2,100))
print(bm2(100,100,1,0.08,0.2,100))

a = []
b = []
c = []
for s0y in np.linspace(50,150,100):
    a.append(s0y)
    b.append(bm(s0y,100,1,0.08,0.2,100))
    c.append(bm2(s0y,100,1,0.08,0.2,100))

plt.plot(a,b)
plt.plot(a,c)
plt.show()

a.clear()
b.clear()
c.clear()

for k0y in np.linspace(50,150,100):
    a.append(k0y)
    b.append(bm(100,k0y,1,0.08,0.2,100))
    c.append(bm2(100,k0y,1,0.08,0.2,100))

plt.plot(a,b)
plt.plot(a,c)
plt.show()

a.clear()
b.clear()
c.clear()

for r0y in np.linspace(0.04,0.12,100):
    a.append(r0y)
    b.append(bm(100,100,1,r0y,0.2,100))
    c.append(bm2(100,100,1,r0y,0.2,100))

plt.plot(a,b)
plt.plot(a,c)
plt.show()

a.clear()
b.clear()
c.clear()

for si0y in np.linspace(0.1,0.3,100):
    a.append(si0y)
    b.append(bm(100,100,1,0.08,si0y,100))
    c.append(bm2(100,100,1,0.08,si0y,100))

plt.plot(a,b)
plt.plot(a,c)
plt.show()

a.clear()
b.clear()
c.clear()

for k in [95,100,105]:
    for m0y in range(10,200):
        a.append(m0y)
        b.append(bm(100,k,1,0.08,0.2,m0y))
        c.append(bm2(100,k,1,0.08,0.2,m0y))
        
    plt.plot(a,b)
    plt.show()
    plt.plot(a,c)
    plt.show()

    a.clear()
    b.clear()
    c.clear()