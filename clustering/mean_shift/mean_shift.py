import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
import pandas as pd

data_base=pd.read_csv("iris.csv")
data_base.drop(['Species'], axis=1, inplace=True)
data_array = data_base.to_numpy()
data_array

mean_s = MeanShift(bandwidth=0.5)
mean_s.fit(data_array)

labels = mean_s.labels_
center = mean_s.cluster_centers_

labels
center

plt.scatter(data_array[:, 2], data_array[:,3], marker='o', s=150, c=labels)
plt.scatter(center[:, 2], center[:, 3], marker='x', s=150, c='r')

plt.show()