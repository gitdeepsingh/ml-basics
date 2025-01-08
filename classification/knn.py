import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values # all but last col i.e purchase
y = dataset.iloc[:, -1].values # purchase

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #random_state: this fixes the seed, at seed 0, randomly picked test datas are always same.


# feature scaling (Standardization)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_train)
print(f'X_train= {X_train}')
print(f'X_test= {X_test}')


