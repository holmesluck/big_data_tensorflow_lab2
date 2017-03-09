# coding=utf-8
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np


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

#initial the variables
m = l
iterationsg = 0
alphag = 0
b = a[:, 0]
c = a[:, 1]
d = a[:, 2]
avx = np.average(b)
avy = np.average(c)
avz = np.average(d)
Sx = np.std(b)
Sy = np.std(c)
Sz = np.std(d)

#Print out the scaled features of the first 10 examples in the dataset
def getExamples():
    print ('Print out the scaled features of the first 10 examples in the dataset')
    for j in range(m):

        x = (b[j]-avx)/Sx

        y = (c[j]-avy)/Sy

        z = d[j]

        if j <= 10:
         print(x, y, z,j)
# get the Jtheta value
f = []
g = []
def getJtheta(Jtheta,i):
    f.append(Jtheta)
    g.append(i)

#calculate the costFunction
def calculatecostFunction(iterations,alpha):
 theta0 = 0
 theta1 = 0
 theta2 = 0
 Jtheta = 0
 cost = 0
 cost1 = 0
 cost2 = 0
 global iterationsg
 global alphag
 iterations = input('please input iterations times:')
 alpha = input('please input alpha value:')
 iterationsg = iterations
 alphag = alpha
 for i in range(iterations):

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

    getJtheta(Jtheta,i)
 predicationValue(theta0,theta1,theta2)
#caculate the predicated price
def predicationValue(theta0,theta1,theta2):
 size = input('please input the testing value of size:')

 room = input('please input the testing value of room:')

 size = (size-avx)/Sx

 room = (room-avy)/Sy

 prediction= theta0+theta1*size+theta2*room

 print ('the predicted value is: '+str(prediction)+''
       'and the theta value is :'+str(theta0)+','+str(theta1)+','+str(theta2))

#print the Jtheta figure
def draw(g,f,iterations,alpha):
 plt.plot(g,f)
 plt.xlim(0, len(g)+1)
 plt.ylim(0, max(f))
 plt.title("costFunction")
 plt.xlabel("iterations")
 plt.ylabel("costFunction(Jtheta)")
 lable1 = 'iteration ='+str(iterations)+' '+'alpha =' + str(alpha)

 plt.legend([lable1] , loc='upper left')

 plt.savefig('/Users/zhangyangzuo/Downloads/capture/test.png')

#input the testing value of the  project1
calculatecostFunction(iterationsg,alphag)
draw(g,f,iterationsg,alphag)
getExamples()
plt.show()

