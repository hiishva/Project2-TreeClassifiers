import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import metrics


def main() -> None:
    df = pd.read_csv("all_data/train_c300_d100.csv", header=None)
    df2 = pd.read_csv("all_data/test_c300_d100.csv", header=None)
    #print(data.head())

    train_labels = df.iloc[:, -1].values.reshape(-1, 1)
    train_data = df.iloc[:, :-1]

    test_labels = df2.iloc[:, -1].values.reshape(-1, 1)
    test_data = df2.iloc[:, :-1]
    print('Train size')
    print(train_data.shape)
    print(train_labels.shape)
    print('Test size')
    print(test_data.shape)
    print(test_labels.shape)

    classifier = DecisionTreeClassifier(criterion='log_loss', max_depth=10)
    classifier.fit(train_data, train_labels)

    test_pred = classifier.predict(test_data)
    print(test_pred.shape)

    print("Accuracy:", metrics.accuracy_score(test_labels, test_pred))

    pass


if __name__ == '__main__':
    main()