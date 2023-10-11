import numpy as np
import glob
import pandas as pd
import sys
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import metrics





'''
generate datasets
create model
tune model
test model
driver function

'''
def ReadFromFolder(directoryPath):
    print('Inside readfromfolder')
    testDataset = [] # All the data values for tests
    trainDataset = [] # all the data values for training
    validDataset = [] # all the data values for validation

    for fileName in glob.glob(directoryPath+'*.csv'):
        print('inside the for loop of the glob')
        if 'test' in fileName:
            x = pd.read_csv(fileName,low_memory=False)
            print("appended to test")
            testDataset.append(x)
        elif 'train' in fileName:
            x = pd.read_csv(fileName,low_memory=False)
            print('appended to train')
            trainDataset.append(x)
        elif 'valid' in fileName:
            x = pd.read_csv(fileName, low_memory=False)
            print('append to valid')
            validDataset.append(x)
    return testDataset, trainDataset, validDataset



def main() -> None:
    directoryName = sys.argv[1]
    print('directory name:{}'.format(directoryName))
    testSet, trainSet, validSet = ReadFromFolder(directoryName)
    print(testSet[1])

    # trainingData = pd.read_csv("all_data/train_c300_d100.csv", header=None)
    # testData = pd.read_csv("all_data/test_c300_d100.csv", header=None)
    # #print(data.head())

    # train_labels = trainingData.iloc[:, -1].values.reshape(-1, 1)
    # train_data = trainingData.iloc[:, :-1]

    # test_labels = testData.iloc[:, -1].values.reshape(-1, 1)
    # test_data = testData.iloc[:, :-1]
    # print('Train size')
    # print(train_data.shape)
    # print(train_labels.shape)
    # print('Test size')
    # print(test_data.shape)
    # print(test_labels.shape)

    # classifier = DecisionTreeClassifier(criterion='log_loss', max_depth=10)
    # classifier.fit(train_data, train_labels)

    

    # test_pred = classifier.predict(test_data)
    # print(test_pred.shape)

    # print("Accuracy:", metrics.accuracy_score(test_labels, test_pred))

    pass


if __name__ == '__main__':
    main()