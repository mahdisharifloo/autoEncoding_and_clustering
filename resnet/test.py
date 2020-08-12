# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
img_path1 = '/home/mahdi/projects/dgkala/server_evoke/evoke_server/simV4.2/similarity_new/data/category-men-shoes/data/4205745.jpg'
with open(img_path1,'rb') as f:
    img_bytes1 = f.read()
Image1 = cv2.imdecode(np.fromstring(img_bytes1,np.uint8),cv2.IMREAD_UNCHANGED)

img=cv2.cvtColor(Image1,cv2.COLOR_BGR2RGB)

vectorized = img.reshape((-1,3))            
vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))

figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()

edges = cv2.Canny(result_image,150,200)
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()