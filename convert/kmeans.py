import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

img = mpimg.imread('image.jpg')
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
K = 20
kmeans = KMeans(n_clusters=K).fit(X)
label = kmeans.predict(X)
print label
img4 = np.zeros_like(X)
# replace each pixel by its center
for k in range(K):
    img4[label == k] = kmeans.cluster_centers_[k]
# reshape and display output image
img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
plt.imshow(img5, interpolation='nearest')
plt.axis('off')
plt.show()
