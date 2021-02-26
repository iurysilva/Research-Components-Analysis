import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from Objects import PCA
from sklearn import neural_network
from sklearn.model_selection import train_test_split
from sklearn import decomposition

width = 90
height = 112
dir = "images"
begin = 6
finish = 6

images_dataset = np.zeros(((finish-begin+1)*10, height*width))
target = []
insert = 0
for directory_name in range(begin, finish+1):
    dir_path = os.path.join(dir, os.listdir(dir)[directory_name])
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path, 0)
        img_resized = cv2.resize(img, (width, height))
        vector = img_resized.ravel()
        images_dataset[insert, :] = vector
        insert += 1
        target += [directory_name]
acuracias = []
#pca = PCA(10)
#pca.fit(images_dataset)
#data_transformed = pca.transform(images_dataset)
pca2 = decomposition.PCA(n_components=10)
data_transformed_2 = pca2.fit_transform(images_dataset)
fig = plt.figure()
'''
for i in range(1, 11):
    fig.add_subplot(2, 5, i)
    plt.imshow(pca2.components_[i-1].reshape(height, width), 'gray')
'''
a = []
for i in pca2.explained_variance_ratio_:
    a.append(i)
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], a, '-ro', linewidth=2)
plt.xticks([1,2,3,4,5,6,7,8,9,10])
'''
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
'''