# Project2-TreeClassifiers
* Download the 15 datasets. Each dataset is divided into three subsets: training, validation, and test set. Datasets are in CSV formats. Each line is a training (or test) example that contains a list of attribute values separated by a comma. The last attribute is the class variable. Assume that all attributes take values from the domain {0,1}
  * Datasets are generated synthetically by randomly sampling solutions and non-solutions from a boolean formula in conjunctive normal form. Randomly generated five formulas having 500 variables and 300, 400, 1000, 1500, and 1800 clauses (where length is 3) respectively, and sampled 100, 1000, and 5000 positive and negative examples from each formula.
  * The naming convention for each of the files is train* test* and valid* denotes the training, test, and validation sets. train_c[i]_d[j].csv where i and j are integers contains training data having j examples generated from the formula that has i clauses. Do not mix and match datasets
* Use ```sklearn.tree.DecisionTreeClassifier``` on the 15 datasets. Use the validation set to tune the parameters. After tuning the parameters mix the training and validation sets, relearn the decision tree using the best parameter setting found via tuning, and report the accuracy and F1-score on the test set. For each dataset, also report the "best parameter settings found via tuning"
* Repeat the experiment from above using: ```sklearn.ensemble.BaggingClassifier``` with "DecisionTreeClassifier" as the base estimator
  * Again use the validation set to tune the parameters, mix training and validation after tuning to learn a new classifier and report
    *  Best parameter setting
    *  Classification accuracy and F1-Score
*  Repeat the experiment using ```sklearn.ensemble.RandomForestClassifier```
*  Repeat the experiment using ```sklearn.ensemble.GradientBoostingClassifier```
*  Record the classification accuracy and F1 score for each dataset and classifier in a table and then answer the following questions using the table:
    * Which classifier (among the four) yields the best overall generalization accuracy/F1 score? Based on your ML knowledge, why do you think the "classifier" achieved the highest overall accuracy/F1 score
    *  What is the impact of increasing the amount of training data on the accuracy/F1 score of each of the four classifiers
    *  What is the impact of increasing the number of features on the accuracy/F1 score of each of the four classifiers
*  Download the MNIST dataset and rescale it using the following code. The dataset has a training set of 60,000 examples and a test set of 10,000 examples where the digits have been centered inside 28 x 28-pixel images.
    ```
    from sklearn.dataset import fetch_openml
    #load the data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=true)
    X = X / 255

    #Rescale the data, use the traditional train/test split
    # (60K: train and 10K: test)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    ```
* Evaluate the four tree eand ensemble classifiers you used on the MNIST dataset (Do not computer F1 score on MNSIT, just classification accuracy). Which classifier among the four yields the best classification accuracy on the MNIST dataset and why?
