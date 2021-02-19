import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import neural_network
from sklearn.model_selection import train_test_split


dir = "images"
dir_to_read = 10

images_dataset = np.zeros((dir_to_read*10, 112*92))
target = []
insert = 0
for directory_name in range(dir_to_read):
    dir_path = os.path.join(dir, os.listdir(dir)[directory_name])
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path, 0)
        img_resized = cv2.resize(img, (112, 92))
        vector = img_resized.ravel()
        images_dataset[insert, :] = vector
        insert += 1
        target += [directory_name]
acuracias = []
pca = PCA(n_components=10)
for i in range(1):
    rede_neural = neural_network.MLPClassifier(hidden_layer_sizes=100, random_state=1, max_iter=200,
                                               activation='logistic')
    data_transformed = pca.fit_transform(images_dataset)
    x_treino, x_teste, y_treino, y_teste = train_test_split(data_transformed, target, test_size=0.3)
    rede_neural.fit(x_treino, y_treino)
    saidas = rede_neural.predict(x_teste)
    print(saidas)
    print(y_teste)
    print('---------')
    print(rede_neural.score(x_teste, y_teste))
    acuracias = np.append(acuracias, rede_neural.score(x_teste, y_teste))
print(acuracias.mean())
