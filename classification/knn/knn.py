#%%
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

print('predicting the test dataset results...')
try:     
    y_pred = classifier.predict(X_test)
    print('writing the results as [[predictedValue groundTruthValue]] into predicted_outputs.txt ...')
    pred_res = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
    with open("predicted_outputs.txt", "w") as file:
        file.write(str(pred_res))
except Exception as e:
    print(f'Error while predicting the test results. Reason= {e}') # gives error if we dont do sc.transform(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)
print(f'accuracy_score = {acc_score}\nconfusion matrix=\n{cm}')
'''
confusion matrix:
[
[TruePositive FalsePositive] 
[FalseNegative TrueNegative]
]

'''
#%%
# Visualising the Training set results
from matplotlib.colors import ListedColormap
print('visualising the training set results....')
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#%%
from matplotlib.colors import ListedColormap
print('visualising the test set results....')
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()