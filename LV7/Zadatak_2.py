import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from PIL import Image as Img
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# ucitaj sliku
img = Img.open("imgs\\test_1.jpg")
img.convert('RGB')
img = np.asarray(img)
# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255
# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

km = KMeans(n_clusters=5)
km.fit(img_array)
labels = km.predict(img_array)
u_labels = np.unique(labels)

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
 
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(img_array_aprox)
    kmeanModel.fit(img_array_aprox)
 
    distortions.append(sum(np.min(cdist(img_array_aprox, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / img_array_aprox.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(img_array_aprox, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / img_array_aprox.shape[0]
    mapping2[k] = kmeanModel.inertia_


for i in u_labels:
    img_array_aprox[labels == i] = km.cluster_centers_[i]
    
plt.figure()
plt.title("Rezultanta slika")
plt.imshow(np.reshape(img_array_aprox, (w, h, d)))
plt.tight_layout()
plt.show()

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()
