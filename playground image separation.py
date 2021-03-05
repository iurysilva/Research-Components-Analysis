import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.decomposition import FastICA

width = 640
height = 427

S1 = cv2.imread('foto.jpg', 0)
S2 = cv2.imread('urso.jpg', 0)
S1 = cv2.resize(S1, (width, height))
S2 = cv2.resize(S2, (width, height))

#blue, green, red = cv2.split(S1)
#blue2, green2, red2 = cv2.split(S2)
#matrices = [blue, green, red, blue2, green2, red2]

A = np.array(([0.06, 0.2], [1/2, 2/3]))

results = []
x1 = A[0, 0] * S1 + A[0, 1] * S2
x2 = A[1, 0] * S1 + A[1, 1] * S2

images_dataset = np.zeros((height * width, 2))
vector1 = x1.ravel()
vector2 = x2.ravel()
images_dataset[:, 0] = vector1
images_dataset[:, 1] = vector2

ica = FastICA(n_components=2)
img = ica.fit_transform(images_dataset)
results.append(img.copy)

figures = [S1, S2, x1, x2, img[:, 0].reshape(height, width), img[:, 1].reshape(height, width)]

fig = plt.figure()
for i in range(1, 7):
    fig.add_subplot(3, 2, i)
    plt.imshow(figures[i-1], 'gray')
