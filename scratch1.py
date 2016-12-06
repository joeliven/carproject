import numpy as np
import matplotlib.pyplot as plt
import os,sys

dir = 'data/raw/line_imgs2'

paths = [os.path.join(dir, x) for x in os.listdir(dir) if x.endswith('.npy')]
# paths = paths[220:230]
print(len(paths))
print('len(paths)')
# imgs = []
# for p in paths:
#     imgs.append(preprocess_image(imread(p)))
# imgs = np.asarray(imgs)
# print('imgs.shape')
# print(imgs.shape)

for img_path in paths:
    img = np.load(img_path)
    plt.imshow(img)
    plt.show()