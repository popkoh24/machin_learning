import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import pandas as pd

data_base=pd.read_csv("iris.csv")

data_base.drop(['Species'], axis=1, inplace=True)
data_array = data_base.to_numpy()
data_array

hirarical = linkage(data_array, method='complete')
# hirarical = linkage(data_array, method='average')
# hirarical = linkage(data_array, method='single')

dendrogram(hirarical)
plt.show()

labels = fcluster(hirarical, 5, criterion='distance')
labels

plt.scatter(data_array[:, 0], data_array[:, 2], c=labels)
plt.show()