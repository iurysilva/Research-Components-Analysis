import numpy as np
from sklearn import decomposition
from sklearn import datasets
from sklearn import neural_network
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


iris = datasets.load_iris()
x = iris.data
y = iris.target
pca = decomposition.PCA(n_components=2)
pca.fit(x)
x_transformed = pca.transform(x)
x_real = x_transformed[:, 0]
y_real = x_transformed[:, 1]

'''
copia_y = []
for valor in range(0, len(y)):
    if y[valor] == 0:
        copia_y.append('red')
    elif y[valor] == 1:
        copia_y.append('blue')
    elif y[valor] == 2:
        copia_y.append('green')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_real, y_real, y, c=copia_y)
ax.set_title('Tipo de Iris')
ax.set_xlabel('Atributo 1')
ax.set_ylabel('Atributo 2')
ax.set_zlabel('Iris')
ax.legend(['flor1', 'flor2', 'flor3'])
plt.show()
'''

x_treino, x_teste, y_treino, y_teste = train_test_split(x_transformed, y, test_size=0.3)
rede_neural = neural_network.MLPClassifier(hidden_layer_sizes=100, random_state=1, max_iter=500,
                                           activation='logistic')
rede_neural.fit(x_treino, y_treino)
saidas = rede_neural.predict(x_teste)
print(saidas)
print(y_teste)
print('---------')
print(rede_neural.score(x_teste, y_teste))
