import math
from math import exp
import numpy as np
import random
import pandas as pd

X =pd.read_csv("1.csv")
print(X.head(5))
N=5  # no.of of fireflies

LB=np.array([1,1,1,1])
UB=np.array([10,10,10,10])
k=2                        #no. of clusters
Z=X.shape[0]               #total no. of data points
print("number of data points :",Z)
F=np.zeros((5,2,4), dtype=np.float64)   #fireflies
J=np.zeros((5), dtype=np.float64)      # objective function initialization



#5  random fireflies are created
C_x = np.random.randint(0, 100, size=k)
C_y = np.random.randint(0, 100, size=k)
Y = np.array(list(zip(C_x, C_y)), dtype=np.float32)
F[0]=Y
print("firefly1")
print(F[0])

C_x = np.random.randint(0, 100, size=k)
C_y = np.random.randint(0, 100, size=k)
Y = np.array(list(zip(C_x, C_y)), dtype=np.float32)
F[1] = Y
print("firefly2")
print(F[1])

C_x = np.random.randint(0, 100, size=k)
C_y = np.random.randint(0, 100, size=k)
Y = np.array(list(zip(C_x, C_y)), dtype=np.float32)
F[2] = Y
print("firefly3")
print(F[2])

C_x = np.random.randint(0, 100, size=k)
C_y = np.random.randint(0, 100, size=k)
Y = np.array(list(zip(C_x, C_y)), dtype=np.float32)
F[3] = Y
print("firefly4")
print(F[3])

C_x = np.random.randint(0, 100, size=k)
C_y = np.random.randint(0, 100, size=k)
Y = np.array(list(zip(C_x, C_y)), dtype=np.float32)
F[4] = Y
print("firefly5")
print(F[4])



arr = np.ones((113, 2), dtype=np.float64)
for j in range(Z):
    for i in range(2):
        dis = math.sqrt(((X[j][0] - F[0][i][0]) ** 2) + ((X[j][1] - F[0][i][1]) ** 2))
        arr[j][i] = dis

#print(arr )
min_dist_clus0 = np.ones((Z), dtype=int)  # stores the nearest cluster no for each data point

for j in range(Z):
    min = 0
    for i in range(2):
        if arr[j][i] < arr[j][min]:
            min = arr[j][i]
    min_dist_clus0[j] = min

#print(min_dist_clus0)

sum=0
for i in range(Z):
    sum=sum+min_dist_clus0[i]

J[0]=sum/Z
print(J[0])


#arr = np.ones((113, 2), dtype=np.float64)
for j in range(Z):
    for i in range(2):
        dis = math.sqrt(((X[j][0] - F[1][i][0]) ** 2) + ((X[j][1] - F[1][i][1]) ** 2))
        arr[j][i] = dis

#print(arr )
#min_dist_clus0 = np.ones((Z), dtype=int)  # stores the nearest cluster no for each data point

for j in range(Z):
    min = 0
    for i in range(2):
        if arr[j][i] < arr[j][min]:
            min = arr[j][i]
    min_dist_clus0[j] = min

#print(min_dist_clus0)

sum=0
for i in range(Z):
    sum=sum+min_dist_clus0[i]

J[1]=sum/Z

print(J[1])

#arr = np.ones((113, 2), dtype=np.float64)
for j in range(Z):
    for i in range(2):
        dis = math.sqrt(((X[j][0] - F[2][i][0]) ** 2) + ((X[j][1] - F[2][i][1]) ** 2))
        arr[j][i] = dis



for j in range(Z):
    min = 0
    for i in range(2):
        if arr[j][i] < arr[j][min]:
            min = arr[j][i]
    min_dist_clus0[j] = min

#print(min_dist_clus0)

sum=0
for i in range(Z):
    sum=sum+min_dist_clus0[i]

J[2]=sum/Z
print(J[2])




arr = np.ones((113, 2), dtype=np.float64)
for j in range(Z):
    for i in range(2):
        dis = math.sqrt(((X[j][0] - F[3][i][0]) ** 2) + ((X[j][1] - F[3][i][1]) ** 2))
        arr[j][i] = dis

#print(arr )
min_dist_clus0 = np.ones((Z), dtype=int)  # stores the nearest cluster no for each data point

for j in range(Z):
    min = 0
    for i in range(2):
        if arr[j][i] < arr[j][min]:
            min = arr[j][i]
    min_dist_clus0[j] = min

#print(min_dist_clus0)

sum=0
for i in range(Z):
    sum=sum+min_dist_clus0[i]

J[3]=sum/Z
print(J[3])



for j in range(Z):
    for i in range(2):
        dis = math.sqrt(((X[j][0] - F[4][i][0]) ** 2) + ((X[j][1] - F[4][i][1]) ** 2))
        arr[j][i] = dis

#print(arr )


for j in range(Z):
    min = 0
    for i in range(2):
        if arr[j][i] < arr[j][min]:
            min = arr[j][i]
    min_dist_clus0[j] = min

#print(min_dist_clus0)

sum=0
for i in range(Z):
    sum=sum+min_dist_clus0[i]

J[4]=sum/Z

print(J[4])


#movement of fireflies
beta = 0
beta_0 = 1
gamma = 1
alpha = 0.5
for k in range(200):
    for i in range(5):
        scale = abs((UB[0] - LB[0])+(UB[1]-UB[1]))
        for j in range(5):
            if(J[i]>J[j]):
                r = abs((F[i][0][0] - F[j][0][0]) + (F[i][0][1] - F[j][0][1])+(F[i][1][0] - F[j][1][0])+(F[i][1][1] - F[j][1][1]))
                r = math.sqrt(r)
                betamin=0.2
                gamma=1.0
                beta0 = 1.0
                beta = (beta0 - betamin) * \
                       math.exp(-gamma * math.pow(r, 2.0)) + betamin
                beta = beta_0 * (exp(-gamma) * (r** 2))
               # print(beta)
                for l in range(2):
                    r = random.uniform(0, 1)
                    tmpf = alpha * (r - 0.5) * scale
                    F[i][l][0] = F[i][l][0] * (1.0 - beta) +F[j][l][0] * beta + tmpf
                    F[i][l][1] = F[i][l][1] * (1.0 - beta) + F[j][l][1] * beta + tmpf
            temp=np.zeros((2,2), dtype=np.float64)
            temp=F[i]
            for l in range(2):
                if temp[l][0] < LB[0]:
                    temp[l][0] = LB[0]
                if temp[l][1] < LB[1]:
                    temp[l][1] = LB[1]
                if temp[l][0] > UB[0]:
                    temp[l][0] = UB[0]
                if temp[l][1] > UB[1]:
                    temp[l][1] = UB[1]

            F[i]=temp
            r = abs((F[i][0][0] - F[j][0][0]) + (F[i][0][1] - F[j][0][1]) + (F[i][1][0] - F[j][1][0]) + (
                        F[i][1][1] - F[j][1][1]))
            r = math.sqrt(r)
            J[i]=J[i]*exp((-gamma)*(r ** 2))

    for i in range(1,5):
        if(J[i-1]<J[i]):
            best=i

print("best firefly ")
print(F[best])






#k=2
#Z = X.shape[0]
#print("number of data points :",Z)

cen_old = np.zeros((2, 2), dtype=np.float64)  # old centroid
cen_new = np.zeros((2, 2), dtype=np.float64)  # new centroid



cen_old = F[best]

print(" firefly best centroid")
print(cen_old)


c1_0 = np.ones((Z, 2), dtype=np.float64)  # 0th cluster
c1_1 = np.ones((Z, 2), dtype=np.float64)  #1st cluster



min_dist_clus0 = np.ones((Z), dtype=int)

arr = np.ones((113, 2), dtype=np.float64) #stores distance between data point and centroid

rot=0
while True:
    rot=rot+1
    sclus_0 = 0  # size of 0th cluster
    sclus_1 = 0  # size of 1st cluster
    for j in range(Z):
        for i in range(2):
            dis = math.sqrt(((X[j][0] - cen_old[i][0]) ** 2) + ((X[j][1] - cen_old[i][1]) ** 2))
            arr[j][i] = dis



    for j in range(Z):
        min = 0
        for i in range(2):
            if arr[j][i] < arr[j][min]:
                min = i
        min_dist_clus0[j] = min


    for i in range(Z):  # clustering
        if min_dist_clus0[i] == 0:
            c1_0[sclus_0][0] = X[i][0]
            c1_0[sclus_0][1] = X[i][1]
            sclus_0 += 1
        if min_dist_clus0[i] == 1:
            c1_1[sclus_1][0] = X[i][0]
            c1_1[sclus_1][1] = X[i][1]
            sclus_1 += 1

    print("cluster 0")
    for i in range(sclus_0):
        print(c1_0[i][0], c1_0[i][1])

    print("cluster 1")
    for i in range(sclus_1):
        print(c1_1[i][0], c1_1[i][1])



    for i in range(sclus_0):  # centroid new calculation
        cen_new[0][0] += c1_0[i][0]
        cen_new[0][1] += c1_0[i][1]

    for i in range(sclus_1):
        cen_new[1][0] += c1_1[i][0]
        cen_new[1][1] += c1_1[i][1]



    if sclus_0 != 0:
        cen_new[0][0] = ((cen_new[0][0]) / sclus_0)
        cen_new[0][1] = ((cen_new[0][1]) / sclus_0)
    if sclus_1 != 0:
        cen_new[1][0] = ((cen_new[0][0]) / sclus_1)
        cen_new[1][1] = ((cen_new[0][1]) / sclus_1)


    print(cen_new)
    if(cen_old[0][0]==cen_new[0][0] and cen_old[0][1]==cen_new[0][1] and cen_old[1][0]==cen_new[1][0] and cen_old[1][1]==cen_new[1][1]):
        break
    else:
        cen_old=cen_new



print("final centroid")
print(cen_new)
print(rot)
