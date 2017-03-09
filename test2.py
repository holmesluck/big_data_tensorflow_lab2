# coding=utf-8
from __future__ import division
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D

#read the data and load it in the array
file = open("/Users/zhangyangzuo/PycharmProjects/untitled/src/house_price_data.txt","r")
lines = file.readlines()
l =len(lines)
for i in range(l):
    lines[i]=lines[i].strip()
    lines[i]=lines[i].strip('[]')
    lines[i]= lines[i].split(",")
a = np.array(lines)
a = a.astype(int)
file.close()


#
theta0 = 0
theta1 = 0
theta2 = 0
Jtheta = 0
b = a[:, 0]
c = a[:, 1]
d = a[:, 2]
e = a[:, 0]

m = l
cost = 0
cost1 = 0
cost2 = 0
alpha = 0.03
iterations = 250

avx = np.average(b)
avy = np.average(c)
avz = np.average(d)
Sx = np.std(b)
Sy = np.std(c)
Sz = np.std(d)

f = []
g = []
fig = plt.figure()


def getJtheta(Jtheta,i):
   # print(i,Jtheta)
   f.append(Jtheta)
   g.append(i)

for i in range(iterations):

    # if i == 2 :
    #     print (theta0,theta1,theta2)

    # e = Jtheta

    # getJtheta(Jtheta, i)

    theta0 = theta0 - cost

    theta1 = theta1 - cost1

    theta2 = theta2 - cost2

    cost = 0

    cost1 = 0

    cost2 = 0

    Jtheta = 0

    for j in range(m):

        x = (b[j]-avx)/Sx

        y = (c[j]-avy)/Sy

        z = d[j]


        cost=cost+alpha*(1/m)*(theta0+theta1*x+theta2*y-z)

        cost1=cost1+alpha*(1/m)*(theta0+theta1*x+theta2*y-z)*x

        cost2=cost2+alpha*(1/m)*(theta0+theta1*x+theta2*y-z)*y

        Jtheta += (0.5/m) * (theta0 + theta1 * x + theta2 * y - z) * (
        theta0 + theta1 * x + theta2 * y - z)
    # print(theta0,theta1,theta2,Jtheta)
    getJtheta(Jtheta,i)

size=(1800-avx)/Sx
room=(3-avy)/Sy

prediction= theta0+theta1*size+theta2*room
print(prediction)




#Compute cost and Gradient Descent for linear regression


# #print the point figure in 3D ax
# mpl.rcParams['legend.fontsize'] = 10
#
#
# ax = fig.gca(projection='3d')
# y = np.linspace(0, max(c), len(c))
# x = np.linspace(0, max(b), len(b))
# z = theta0+theta1*x+theta2*y
# for i in range(l):
#     xs = a[i][0]
#     ys = a[i][1]
#     zs = a[i][2]
#     ax.scatter(xs, ys, zs, c='r', marker='o')
# ax.plot(x, y, z, label="iterations number ")
#
# ax.set_zlabel('price(USD)')
# ax.set_ylabel('num of Bedroom' )
# ax.set_xlabel('Size (square feet)')
# plt.savefig('/Users/zhangyangzuo/Downloads/capture/test1.png')

#print the Jtheta figure
plt.plot(g,f)
plt.xlim(0, len(g)+1)
plt.ylim(0,max(f))
plt.title("costFunction")
plt.xlabel("iterations")
plt.ylabel("costFunction(Jtheta)")
plt.savefig('/Users/zhangyangzuo/Downloads/capture/test1.png')
plt.show()