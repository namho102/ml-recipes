#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
from PIL import Image
import random
from sklearn.cluster import KMeans
import numpy as np

# class KMeans:

#     def __init__(self, arr, k=8):
#         self.k = k
#         self.arr = arr
#         self.size = len(self.arr)
#         self.labels = [0] * self.size
#         self.centroids = self.random_centeroids()

#     def random_centeroids(self):
#         centroids = []
#         for i in range(self.k):
#             num = random.randint(0, self.size - 1)
#             print num
#             centroids.append(self.arr[num])
#         return centroids

#     def set_centeroid(self, point, x):
#     	print self.centroids[0]
#         min = self.distance(point, self.centroids[0])
#         for i in range(self.k):
#             if self.distance(point, self.centroids[i]) < min:
#                 self.labels[x] = i

#     def reset_centeroid(self):
#         sum = [[0, 0, 0]] * self.k
#         count = [0] * self.k
#         for i in range(self.size):
#         	label = self.labels[i]
#         	count[label] += 1
#         	for j in range(3):
#         		sum[label][j] += self.arr[i][j]
                

#         # print sum          
#        	for i in range(self.k):
#             for j in range(3):
#             	# print sum[i][j]  
#             	if count[i] is not 0:
#             		sum[i][j] /= count[i]

#         # print "After:", sum    		
#         new_centroids = [()]*self.k

#         for i in range(self.k):
#         	new_centroids[i] = tuple(sum[i])

#         # print new_centroids	
#         # return new_centroids
        

#     def calculate(self):
#         self.centroids = self.random_centeroids()
#         for i in range(500):
#             for j in range(self.size):
#                 self.set_centeroid(self.arr[j], j)
#             self.centroids = self.reset_centeroid()
#             # print self.labels


#     def distance(self, a, b):
#         (a1, b1, c1) = a
#         (a2, b2, c2) = b
#         return math.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2 + (c1
#                          - c2) ** 2)

def distance(a, b):
	return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - a[2]) ** 2)



im = Image.open('image.jpg')
rgb_im = im.convert('RGB')
(width, height) = im.size
k = 8

colors = []
for x in range(width):
    for y in range(height):
        color = rgb_im.getpixel((x, y))
        colors.append(color)


for color in colors:
	color = list(color)

# X = np.array(colors)
# kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
# centers = kmeans.cluster_centers_
centers = [[ 173, 122,   95],
	 [  92,   45,   29],
	 [ 208,  198,  191],
	 [ 120,   96,   63],
	 [ 183,  159,  135],
	 [  17,   11,   10],
	 [ 234,  103,   62],
	 [  48,   57,   49]]

for center in centers:
	for i in range(3):
		center[i] = int(center[i])

# print centers		


data = []

# for i in range(len(colors)):
# 	if i % width is 0:
# 		row = []
# 	min = distance(colors[i], centers[0])
# 	new_point = centers[0]
# 	for j in range(k):
# 		new_point = centers[j]
# 		if distance(colors[i], centers[j]) < min:
# 			new_point = centers[j]

# 	row.append(new_point)
# 	if i % width is 2:
# 		data.append(row)			


for i in range(len(colors)):
	if i % width is 0:
		row = []

	row.append(colors[i])
	
	if i % width is 2:
		data.append(row)		

data = np.array(data)
# print len(data[0])
print data[0][0]

img = Image.fromarray(data, 'RGB')
img.save('my.png')
img.show()




			