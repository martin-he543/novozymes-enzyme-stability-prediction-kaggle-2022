import numpy as np
import matplotlib.pyplot as plt

titleFont = {'fontname': 'Kinnari', 'size': 13}
axesFont = {'fontname': 'Kinnari', 'size': 9}
ticksFont = {'fontname': 'SF Mono', 'size': 7}
errorStyle = {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle = {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle = {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1}
histStyle = {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}

pH, tm = np.genfromtxt("sample_data/training.csv",skip_header=1,unpack=True,delimiter=",")

plt.plot(pH,tm,'x')
plt.xlabel("pH", **axesFont)
plt.ylabel("Thermostability", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.title("pH against Thermostability Values", **titleFont)
# plt.savefig('Task 12.jpg', dpi=1000)
plt.show()
