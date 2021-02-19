import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

image = cv2.imread('urso.jpg')
print(image)
blue, green, red = cv2.split(image)
print('-----')
print(green)
print('-----')
pca = PCA(n_components=10)

red_transformed = pca.fit_transform(red)
red_inverted = pca.inverse_transform(red_transformed)

green_transformed = pca.fit_transform(green)
green_inverted = pca.inverse_transform(green_transformed)

blue_transformed = pca.fit_transform(blue)
blue_inverted = pca.inverse_transform(blue_transformed)

descompressed = np.dstack((red_inverted, green_inverted, blue_inverted)).astype(np.uint8)
compressed = np.dstack((red_transformed, green_transformed, blue_transformed)).astype(np.uint8)
plt.imshow(descompressed)
#img = PIL.Image.fromarray(compressed)
#img.save('test.jpg')
#img2 = PIL.Image.fromarray(transformed)
#img2.save('test2.jpg')
