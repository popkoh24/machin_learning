import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import *
from sklearn import model_selection
from sklearn.model_selection import *
import seaborn as sns

#--------read data---------
svmdata = pd.read_csv('diabetes.csv')

#-----create and shape data-----
df = pd.DataFrame(svmdata)
df.shape

#-----delet invalid column and set targets-----
x = df.drop('Outcome',axis=1)
y = df['Outcome']

#-----algorithme and fit----
# clf = SVC(kernel='linear')
# clf = SVC(kernel='poly')
clf = SVC(kernel='rbf')

clf.fit(x, y)

#-----train test-----
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3)

#---------
y_pred = clf.predict(x_test)
y_test == y_pred

#----------tp fn tn fp---------
confusion_matrix(y_test, y_pred)

#
print(classification_report(y_test, y_pred))

#
arr = confusion_matrix(y_test, y_pred)
tn,fp,fn,tp = arr.reshape(-1)

#
scores = cross_val_score(clf, x, y,cv=5, scoring='accuracy')

scores.mean()

