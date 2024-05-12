import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#-------read and creat data--------
mlp = pd.read_csv('diabetes.csv')
df = pd.DataFrame(mlp)

#------delet in vain columns------
x = df.drop('SkinThickness',axis=1)
x = x.drop('Outcome',axis=1)

#-----label(target)------
y = df['Outcome']

#-----train test-----
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3)

#------agorithme---------
my_mlp = MLPClassifier(hidden_layer_sizes=40, max_iter=103, alpha=1e-4)

#-----fit data-----
my_mlp.fit(x_train, y_train)

#----------
y_pred = my_mlp.predict(x_test)
y_pred == y_test

#-----tt tf ft ff------
confusion_matrix(y_test, y_pred)

#-----score-----
my_mlp.score(x_train, y_train)
my_mlp.score(x_test, y_test)

score = cross_val_score(my_mlp, x, y, cv=3)
score.mean()