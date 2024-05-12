import numpy as np
import pandas as pd
from sklearn import preprocessing

#-------read data--------
pdata = pd.read_csv('diabetes.csv')

#-------delet in valid columns--------
pdata.drop('SkinThickness', axis=1, inplace=True)

#----algorithme-----
from sklearn.impute import SimpleImputer

#-----fit-----
i = SimpleImputer(missing_values=np.nan, strategy='mean')#5
i.fit(pdata)

#------------
new_df = i.transform(pdata)