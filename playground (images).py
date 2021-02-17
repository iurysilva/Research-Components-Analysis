import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

image = cv2.imread('urso.jpg')
image = image.astype(float)
image_2 = np.zeros((image.shape[0], image.shape[1], 1))
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        new_value = image[i][j][0] * 0.299 + image[i][j][1] * 0.587 + image[i][j][2] * 0.114
        image_2[i][j][0] = np.copy(new_value)
print(image_2)
blue, green, red = cv2.split(image)
pca = PCA(n_components=10)

red_transformed = pca.fit_transform(red)
red_inverted = pca.inverse_transform(red_transformed)

green_transformed = pca.fit_transform(green)
green_inverted = pca.inverse_transform(green_transformed)

blue_transformed = pca.fit_transform(blue)
blue_inverted = pca.inverse_transform(blue_transformed)

descompressed = np.dstack((red_inverted, green_inverted, blue_inverted)).astype(np.uint8)
compressed = np.dstack((red_transformed, green_transformed, blue_transformed)).astype(np.uint8)
plt.imshow(image_2)
#img = PIL.Image.fromarray(compressed)
#img.save('test.jpg')
#img2 = PIL.Image.fromarray(transformed)
#img2.save('test2.jpg')
