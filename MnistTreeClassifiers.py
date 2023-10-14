from sklearn import metrics
from sklearn.datasets import fetch_openml
import pandas as pd
import sys
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def loadMNIST():
    #load the data from open ML
    X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X / 255

    X_train, X_test = X[:60000], X[60000:]
    label_train, label_test = y[:60000], y[60000:]
    return X_test,X_train,label_train,label_test

def DecisionTree(X_test, X_train, label_train, label_test):
    dtClassifier = DecisionTreeClassifier()
    dtClassifier.fit(X_train,label_train)
    preds = dtClassifier.predict(X_test)
    print('Decision Tree Accuracy: {}'.format(metrics.accuracy_score(label_test, preds)))
    return

def BaggingClassifiers(X_test, X_train, label_train, label_test):
    bgClassifier = BaggingClassifier()
    bgClassifier.fit(X_train,label_train)
    pred = bgClassifier.predict(X_test)
    print('Bagging Classifier Accuracy: {}'.format(metrics.accuracy_score(label_test,pred)))

def RandomForest(X_test, X_train, label_train, label_test):
    rfClassifier = RandomForestClassifier()
    rfClassifier.fit(X_train,label_train)
    pred = rfClassifier.predict(X_test)
    print('Random Forest Classifier Accuracy: {}'.format(metrics.accuracy_score(label_test,pred)))

def GradientBoosting(X_test, X_train, label_train, label_test):
    gbClassifier = GradientBoostingClassifier()
    gbClassifier.fit(X_train,label_train)
    pred = gbClassifier.predict(X_test)
    print('Gradient Boosting Classifier Accuracy: {}'.format(metrics.accuracy_score(label_test,pred)))

warnings.filterwarnings("ignore", category=FutureWarning)
print('about to load data')
X_test, X_train, label_train, label_test = loadMNIST()
print('loaded data')
print('about to call Decision Tree classifier')
DecisionTree(X_test, X_train, label_train, label_test)
print('called decision tree classifier')
print('about to call bagging classifiers')
BaggingClassifiers(X_test, X_train, label_train, label_test)
print('called bagging classifer')
print('about to call random forest classifier')
RandomForest(X_test, X_train, label_train, label_test)
print('called random forest classifier')
print('about to call gradient boosting classifier')
GradientBoosting(X_test, X_train, label_train, label_test)

