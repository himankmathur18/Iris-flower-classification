# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, names=columns)

# Display the first few rows of the dataset
print(iris.head())

# Summary statistics
print(iris.describe())

# Check for missing values
print(iris.isnull().sum())

# Visualize the data
sns.pairplot(iris, hue='species')
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train) # type: ignore
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

from sklearn.model_selection import cross_val_score
import numpy as np

# Cross-validation for Logistic Regression
cv_scores_lr = cross_val_score(lr, X, y, cv=5)
print("Logistic Regression CV Scores:", cv_scores_lr)
print("Logistic Regression CV Mean Score:", np.mean(cv_scores_lr))

# Cross-validation for KNN
cv_scores_knn = cross_val_score(knn, X, y, cv=5)
print("KNN CV Scores:", cv_scores_knn)
print("KNN CV Mean Score:", np.mean(cv_scores_knn))

# Cross-validation for Decision Tree
cv_scores_dt = cross_val_score(dt, X, y, cv=5)
print("Decision Tree CV Scores:", cv_scores_dt)
print("Decision Tree CV Mean Score:", np.mean(cv_scores_dt))

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Neural Network
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10)

# Evaluate the neural network
_, accuracy = model.evaluate(X_test, y_test)
print('Neural Network Accuracy: %.2f' % (accuracy*100))
