# scikit-learn logistic regression
import os

import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # load dataset
    print('step 1: load dataset')
    data = pd.read_csv('./datasets/iris.csv', sep=',')

    print('step 2: select target column')
    X = data[['a2', 'a3']]
    y = data['target']

    print('step 3: split data into training and test sets')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    print('train ', X_train_std.shape)
    print('test ', X_test_std.shape)

    # change regularization rate and you will likely get a different accuracy.
    reg = 10.01

    print('step 4: select and train model')
    # train a logistic regression model on the training set
    clf1 = LogisticRegression(C=1 / reg).fit(X_train_std, y_train)
    print(clf1)

    # evaluate the test set
    print('step 5: evaluate model')
    y_pred = clf1.predict(X_test_std)

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred), '\n')
    print('r2_score: {}'.format(r2_score(y_test, y_pred)))
    print('mean_squared_error: {}'.format(mean_squared_error(y_test, y_pred)))

    accuracy = clf1.score(X_test_std, y_test)
    print("Accuracy is {}".format(accuracy))

    directory = './output'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save model to a .pkl file
    joblib.dump(clf1, './output/model.pkl')

    # load model again from .pkl file
    clf2 = joblib.load('./output/model.pkl')
    accuracy = clf2.score(X_test_std, y_test)
    print("Accuracy is {}".format(accuracy))


if __name__ == '__main__':
    # execute only if run as a script
    main()
