import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd


data_base=pd.read_csv("iris.csv")

data_base.drop(['Species'], axis=1, inplace=True)


data_array = data_base.to_numpy()
data_array


ds = DBSCAN(eps=1, min_samples=5)
ds.fit(data_array)
labels = ds.labels_
labels

plt.scatter(data_array[:,0], data_array[:,2], c=labels)
plt.show()