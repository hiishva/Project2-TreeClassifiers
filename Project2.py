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


def ReadFromFolder(directoryPath):
    #print('Inside readfromfolder')
    testDataset = [] # All the data values for tests
    trainDataset = [] # all the data values for training
    validDataset = [] # all the data values for validation

    for fileName in sorted(glob.glob(directoryPath+'*.csv')):
        if 'test' in fileName:
            x = pd.read_csv(fileName,low_memory=False,header=None)
            testDataset.append(x)
        elif 'train' in fileName:
            x = pd.read_csv(fileName,low_memory=False,header=None)
            trainDataset.append(x)
        elif 'valid' in fileName:
            x = pd.read_csv(fileName, low_memory=False,header=None)
            validDataset.append(x)
    return testDataset, trainDataset, validDataset

## DECISION TREE CLASSIFIER ##
def DecisionTrees(trainData, trainLabs, testData, testLabs,validData, validLabs):
    print('Train Size: {}'.format(trainData.shape))
    print('Valid Size: {}'.format(validData.shape))
    print('Test Size: {}'.format(testData.shape))
    
    #Decision Tree Classifier training on training data
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
    
    #Re-training with the best parameters on the combination training and validation data
    BestParamDecTreeClassifier = DecisionTreeClassifier(**bestParams)
    BestParamDecTreeClassifier.fit(np.concatenate([trainData, validData]), np.concatenate([trainLabs, validLabs])) #combine the training and validation data


    testPredB = BestParamDecTreeClassifier.predict(testData) #Run the test

    #Print out the accuracy and the F-1 Score
    print('Accuracy of Test: {}'.format(metrics.accuracy_score(testLabs, testPredB)))
    print('F1 Score of Test: {}'.format(metrics.f1_score(testLabs, testPredB)))
    print('-----------')

    return

## BAGGING CLASSIFIER WITH BASE ESTIMATOR OF DECISIONTREE CLASSIFIER ##
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
                'bootstrap_features': [True, False]}

    gridSearchBag = GridSearchCV(bagClass, paramGridBag, cv=5, scoring='accuracy')
    gridSearchBag.fit(validData,validLabel)
    baggingBestParams = gridSearchBag.best_params_
    print('The best parameters: {}'.format(baggingBestParams))
    bestBagging = BaggingClassifier(**baggingBestParams)

    # Re-training with the best parameters on combined training and validation data 
    bestBagging.fit(np.concatenate([trainData, validData]), np.concatenate([trainLabel, validLabel])) #combine the training and validation data
    testPredB = bestBagging.predict(testData) #Run the test

    # Print out the accuracy and the F-1 Score
    print('Accuracy of Test: {}'.format(metrics.accuracy_score(testLabel, testPredB)))
    print('F1 Score of Test: {}'.format(metrics.f1_score(testLabel, testPredB)))
    print('-----------')


def RandomForest(trainData, trainLabel, testData, testLabel, validData, validLabel):
    print('Training Size: {}'.format(trainData.shape))
    print('Test Size: {}'.format(testData.shape))
    print('Valid Size: {}'.format(validData.shape))

    #Create Random Forest and train on training data
    rfClassifier = RandomForestClassifier() #Base estimator is the decisionTree classifier by default
    rfClassifier.fit(trainData,trainLabel)
    #testPred = rfClassifier.predict(testData)
    #print('Test pred size {}'.format(testPred.shape))
    #print('Accuracy b4 tuning: {}'.format(metrics.accuracy_score(testLabel, testPred)))

    # Parameter tuning
    paramGrid = {
                'criterion':['gini','entropy','log_loss'],
                'n_estimators': [100,250,500],
                'max_features':['sqrt','log2'],
                'max_depth': [10,100, None],
                'min_samples_split':[2,5,10],
                'min_samples_leaf':[1,2,4]}
    
    rfGridSearch = GridSearchCV(rfClassifier, paramGrid, cv = 2,scoring='accuracy', verbose=2)
    rfGridSearch.fit(validData,validLabel)
    rfBestParams = rfGridSearch.best_params_
    print('The best parameters: {}'.format(rfBestParams))
    bestRF = RandomForestClassifier(**rfBestParams)

    # Re-train data with the best parameters on combined training and validation data 
    bestRF.fit(np.concatenate([trainData,validData]), np.concatenate([trainLabel, validLabel]))
    testPredB = bestRF.predict(testData)

    #Print out accuracy and the score
    print('Accuracy of Test: {}'.format(metrics.accuracy_score(testLabel, testPredB)))
    print('F-1 Score of Test: {}'.format(metrics.f1_score(testLabel,testPredB)))
    print('-----------')
    return


## GRADIENT BOOSTING ##
def GradientBoosting(trainData, trainLabel, testData, testLabel, validData, validLabel):
    print('Training Size: {}'.format(trainData.shape))
    print('Test Size: {}'.format(testData.shape))
    print('Valid Size: {}'.format(validData.shape))
    return

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
        #BaggingClassifiers(trainData, trainLabel, testData, testLabel, validData, validLabel)

        #Random Forest
        RandomForest(trainData, trainLabel, testData, testLabel, validData, validLabel)
    pass


if __name__ == '__main__':
    main()







