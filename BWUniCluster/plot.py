import matplotlib.pyplot as plt
import numpy as np

u = np.load('ux.npy')
v = np.load('uy.npy')

L = u[0].size

fig, ax = plt.subplots()

x = np.arange(L)
y = np.arange(L)
plt.gca().invert_yaxis()

# plt.suptitle("Sliding Lid on BWUniCluster with 144 processes", fontweight="bold")

norm = plt.Normalize(0, 0.1)
ax.streamplot(x, y, v, u, color=np.sqrt(u**2 + v**2) if u.sum() and v.sum() else None, norm=norm, density=2.0)  

plt.show()