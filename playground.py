from sklearn import decomposition
from sklearn import datasets
from sklearn import neural_network
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
x = iris.data
y = iris.target


pca = decomposition.PCA(n_components=1)
pca.fit(x)
x = pca.transform(x)
x_treino, x_teste, y_treino, t_teste = train_test_split(x, y, test_size=0.3)
rede_neural = neural_network.MLPClassifier(hidden_layer_sizes=2)




'''
iris = datasets.load_iris()
atributos, classes = iris.data, iris.target
print(atributos)
'''