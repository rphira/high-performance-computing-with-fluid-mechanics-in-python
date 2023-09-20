import matplotlib.pyplot as plt
import numpy as np

'''300x300'''
n900 = 161.56
n784 = 166.58
n400 = 286.73
n324 = 273.85
n256 = 292.76
n196 = 261.83
n144 = 131.75
n100 = 121.09
n81 = 128.54
n64 = 136
n36 = 107.42
n16 = 81.18
n9 = 51.03
n4 = 17.26

'''500x500'''
l900 = 412.02
l784 = 427.65
l400 = 678.35 
l324 = 633.05
l256 = 635.96 
l196 = 545.51 
l144 = 518.18 
l100 = 242.04 
l81 = 204.69
l64 = 227.95
l36 = 160.45
l16 = 94.52
l4 = 16.39

'''plotting'''
fig, ax = plt.subplots()
ncpu = np.array([4, 9, 16, 36, 64, 81, 100, 144, 196, 256, 324, 400, 784, 900])
x = np.array([n4, n9, n16, n36, n64, n81, n100, n144, n196, n256, n324, n400, n784, n900])
y = np.array([l4, None, l16, l36, l64, l81, l100, l144, l196, l256, l324, l400, l784, l900])
ax.plot(ncpu, x, marker='o', label="300x300")
ax.plot(ncpu, y, marker='o', label="500x500")
# ax = plt.gca()
# ax.set_xlim([-10, 1000])
# ax.set_ylim([0, 1000])
plt.xlabel("No. of processes")
plt.ylabel("Million lattice updates per second (MLUPS)")
plt.legend()
plt.suptitle("MLUPS vs. No. of Processes", fontweight="bold")
plt.show()