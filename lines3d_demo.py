import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

file = open("/Users/zhangyangzuo/PycharmProjects/untitled/src/house_price_data.txt","r")
lines = file.readlines()

l = len(lines)
for i in range(l):
    lines[i]=lines[i].strip()
    lines[i]=lines[i].strip('[]')
    lines[i]= lines[i].split(",")
a = np.array(lines)
a = a.astype(int)
print (a)
file.close()
mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
# for i in range(l):
#     x = a[i][0]
#     y = a[i][1]
#     z = a[i][2]
#     print(x)
#     print(y)
#     print(z)
ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()
