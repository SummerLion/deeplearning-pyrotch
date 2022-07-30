import numpy as np
import matplotlib.pyplot as plt




x = np.linspace(-10,10)
y = np.sin(x)
plt.style.use('dark_background')
plt.plot(x,y)
plt.xlabel('time')
plt.ylabel('speed')

plt.show()