import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('reading the dataset.')
dataset = pd.read_csv('Social_Network_Ads.csv')

print('setting X inputs and y target.')
X = dataset.iloc[:, :-1].values # all but last col i.e purchase
y = dataset.iloc[:, -1].values # purchase

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
print('splitting the dataset into the training and test sets.')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #random_state: this fixes the seed, at seed 0, randomly picked test datas are always same.


# feature scaling (Standardization)
from sklearn.preprocessing import StandardScaler
print('standardizing the training set for feature scaling.')
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(f'X_train= {X_train}')
# print(f'X_test= {X_test}')

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
print('training the knn model on training dataset.')
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) # at p=2, minkowski distance represents euclidean distance
classifier.fit(X_train, y_train)

# predicting a new point
predicted_res = classifier.predict(sc.transform([[30, 90000]]))
print('\n')
print('predicting a new data point [30, 90000]. Result= ', predicted_res)
print('\n')

print('predicting the test dataset results...')
try:     
    y_pred = classifier.predict(X_test)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
except Exception as e:
    print(f'Error while predicting the test results. Reason= {e}') # gives error if we dont do sc.transform(X_test)
