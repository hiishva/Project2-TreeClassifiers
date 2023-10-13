import numpy as np
import glob
import pandas as pd
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics





'''
generate datasets
create model
tune model
test model
driver function

'''
def ReadFromFolder(directoryPath):
    #print('Inside readfromfolder')
    testDataset = [] # All the data values for tests
    trainDataset = [] # all the data values for training
    validDataset = [] # all the data values for validation

    for fileName in sorted(glob.glob(directoryPath+'*.csv')):
        #print('inside the for loop of the glob')
        if 'test' in fileName:
            x = pd.read_csv(fileName,low_memory=False,header=None)
            #print("{} appended to test".format(fileName))
            testDataset.append(x)
        elif 'train' in fileName:
            x = pd.read_csv(fileName,low_memory=False,header=None)
            #print('{} appended to train'.format(fileName))
            trainDataset.append(x)
        elif 'valid' in fileName:
            x = pd.read_csv(fileName, low_memory=False,header=None)
            #print('{} append to valid'.format(fileName))
            validDataset.append(x)
    return testDataset, trainDataset, validDataset

def DecisionTrees(trainData, trainLabs, testData, testLabs,validData, validLabs):
    print('Train Size: {}'.format(trainData.shape))
    print('Valid Size: {}'.format(validData.shape))
    print('Test Size: {}'.format(testData.shape))
    
    #Decision Tree Classifier training on trainin data
    decTreeClassifier = DecisionTreeClassifier()
    decTreeClassifier.fit(trainData,trainLabs)
    
    #Parameter tuning
    paramGrid = {'criterion':['gini','entropy','log_loss'],
                'splitter':['best','random'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split':[2, 5, 10],
                'min_samples_leaf':[1, 2, 4]}
    gridSearchCV = GridSearchCV(decTreeClassifier, paramGrid, cv=5, scoring='accuracy')
    gridSearchCV.fit(validData,validLabs)
    bestParams = gridSearchCV.best_params_ #Get the best parameters
    print('The best parameters are: {}'.format(bestParams)) 
    
    #Re-training with the best parameters
    BestParamDecTreeClassifier = DecisionTreeClassifier(**bestParams)
    BestParamDecTreeClassifier.fit(np.concatenate([trainData, validData]), np.concatenate([trainLabs, validLabs])) #combine the training and validation data


    testPredB = BestParamDecTreeClassifier.predict(testData) #Run the test

    #Print out the accuracy and the F-1 Score
    print('Accuracy of Test: {}'.format(metrics.accuracy_score(testLabs, testPredB)))
    print('F1 Score of Test: {}'.format(metrics.f1_score(testLabs, testPredB)))
    print('-----------')

    return

def BaggingClassifiers(trainData,trainLabel,testData, testLabel, validData, validLabel):
    print('Train Size: {}'.format(trainData.shape))
    print('Valid Size: {}'.format(validData.shape))
    print('Test Size: {}'.format(testData.shape))

    
    # Create and the train the bagging classfier
    bagClass = BaggingClassifier() #Default base estimator is DecisionTreeClassfier

    bagClass.fit(trainData,trainLabel)
    testPred = bagClass.predict(testData)
    print('Test Pred Size: {}'.format(testPred.shape))
    print('Accuracy of Test b4 tuning: {}'.format(metrics.accuracy_score(testLabel, testPred)))
    
    # Parameter tuning
    paramGridBag = {
            'n_estimators': [10,100,1000],
            'max_samples' :[1, 2, 5],
            'max_features':[1, 10, 100],
            'bootstrap': [True, False],
            'bootstrap_features': [True, False]
    }

    gridSearchBag = GridSearchCV(bagClass, paramGridBag, cv=5, scoring='accuracy')
    gridSearchBag.fit(validData,validLabel)
    baggingBestParams = gridSearchBag.best_params_
    print('The best parameters: {}'.format(baggingBestParams))
    bestBagging = BaggingClassifier(**baggingBestParams)

    # Re-training with the best parameters
    bestBagging.fit(np.concatenate([trainData, validData]), np.concatenate([trainLabel, validLabel])) #combine the training and validation data
    testPredB = bestBagging.predict(testData) #Run the test

    # Print out the accuracy and the F-1 Score
    print('Accuracy of Test: {}'.format(metrics.accuracy_score(testLabel, testPredB)))
    print('F1 Score of Test: {}'.format(metrics.f1_score(testLabel, testPredB)))
    print('-----------')

def main() -> None:
    directoryName = sys.argv[1]
    print('directory name:{}'.format(directoryName))
    testSet, trainSet, validSet = ReadFromFolder(directoryName)
    #print(testSet[1])

    #Create labels
    for training,testing,validating in zip(trainSet, testSet,validSet) :
        trainLabel = training.iloc[:, -1].values.reshape(-1,)
        trainData = training.iloc[:,:-1]
        testLabel = testing.iloc[:,-1].values.reshape(-1,)
        testData = testing.iloc[:,:-1]
        validLabel = validating.iloc[:, -1].values.reshape(-1,)
        validData = validating.iloc[:, :-1]

        #Decision Tree Classifier
        #DecisionTrees(trainData, trainLabel, testData, testLabel,validData, validLabel)

        #Bagging Classifier
        BaggingClassifiers(trainData, trainLabel, testData, testLabel, validData, validLabel)
    
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







