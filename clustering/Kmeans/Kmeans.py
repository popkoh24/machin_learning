import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd


data_base=pd.read_csv("iris.csv")

data_base.drop(['Species'], axis=1, inplace=True)
data_array = data_base.to_numpy()
data_array


km = KMeans(n_clusters=2)
km.fit(data_base)

labels = km.labels_
labels

center = km.cluster_centers_
center


plt.scatter(data_array[:,2], data_array[:,3], c=labels)
plt.scatter(center[:,2], center[:,3], marker='X', s=150)
plt.show()