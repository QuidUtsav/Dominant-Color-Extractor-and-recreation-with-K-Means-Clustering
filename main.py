import pandas as pd
from scipy.cluster.vq import kmeans,whiten,vq
from matplotlib import image as img
import numpy as np
from matplotlib import pyplot as plt
img_f = img.imread('image.jpeg')
import math

r=[]
g=[]
b=[]
for row in img_f:
    for temp_r,temp_g,temp_b in row:
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)
r=np.array(r)
g=np.array(g)
b=np.array(b)
r_std = np.std(r)
g_std = np.std(g)
b_std = np.std(b)
scaled_r = r / r_std
scaled_g = g / g_std
scaled_b = b / b_std

df=pd.DataFrame({'red':scaled_r,'green':scaled_g,'blue':scaled_b})


cluster_centers,_=kmeans(df.values,4)
df['labels'],_ = vq(df.values,cluster_centers)

result = cluster_centers[df['labels']]

result = np.array(result)
result[:,0] = result[:,0]*r_std
result[:,1] = result[:,1]*g_std
result[:,2] = result[:,2]*b_std

result_int = np.clip(result, 0, 255).round().astype(np.uint8)

h,w,_ = img_f.shape
result_int = result_int.reshape(h,w,3)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_f)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Quantized (Dominant Colors Only)")
plt.imshow(result_int)   
plt.axis('off')

plt.show()