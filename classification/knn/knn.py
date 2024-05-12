import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn import datasets

#------read data-------
datasets = datasets.load_iris()

#------shape data-------
datasets.data.shape

#------targes names-------
datasets.target_names

#------show data set as frame-------
df = pd.DataFrame(datasets.data, columns = datasets.feature_names)

#------creat label-------
df['target'] = datasets.target

#------show-------
pd.plotting.scatter_matrix(df, c=df.target, s=150)

#------data and label-------
x = datasets.data
y = datasets.target

#------import algorithme-------
from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier(n_neighbors=3
, metric='minkowski', p=2)
#------seprate train and test-------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#------fit-----
Knn.fit(x_train, y_train)
#----%100-----
Knn.score(x_test, y_test)
#-------------
neighbors = np.arange(1, 30)
train_acc = np.empty(len(neighbors))
test_acc = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    Knn_model = KNeighborsClassifier(n_neighbors=k)
    Knn_model.fit(x_train, y_train)
    train_acc[i] = Knn_model.score(x_train, y_train)
    test_acc[i] = Knn_model.score(x_test, y_test)
    
#------show in x&y
plt.plot(neighbors, train_acc, label='Train Accuracy')
plt.plot(neighbors, test_acc, label='Test Accuracy')
plt.legend()
plt.show()
