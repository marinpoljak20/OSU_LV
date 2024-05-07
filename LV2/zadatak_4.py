import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50, 50))
white = np.ones((50, 50))

first = np.hstack([black, white])
second = np.hstack([white, black])

img = np.vstack([first, second])

plt.figure()
plt.imshow(img, cmap="gray")
plt.show()