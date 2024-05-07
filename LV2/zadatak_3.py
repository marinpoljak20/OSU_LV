import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("LV2\\road.jpg")
img = img[:,:,0].copy()

plt.figure()

#a
plt.imshow(img, cmap="gray", alpha=0.5)
plt.show()

#b
plt.imshow(img[:,160:320:], cmap="gray")
plt.show()

#c
plt.imshow(np.rot90(img), cmap="gray")
plt.show()

#d
plt.imshow(np.flip(img, axis=1), cmap="gray")
plt.show()