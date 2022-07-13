import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt


def read_data(path, y_column_name):
    data = pd.read_csv(path)
    return data.drop([y_column_name], axis=1), data[[y_column_name]]


def plot(X, y, id_column, max_depth=25, n_splits=10):
    columns = ['Max depth', 'Train accuracy with id', 'Train accuracy without id',
               'Test accuracy with id', 'Test accuracy without id']
    table_cross = pd.DataFrame(columns=columns)
    table_holdout = pd.DataFrame(columns=columns)

    # Place the ID column in the 0th column
    X = pd.concat([X[[id_column]], X.drop(id_column, axis=1)], ignore_index=True, axis=1)
    X = X.to_numpy()
    y = y.to_numpy()

    for i in range(1, max_depth+1):
        row_cross = np.zeros(5)
        row_cross[0] = i

        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = y[train_index], y[test_index]
            curr1, curr3 = calculate_accuracy(X_train, X_test, Y_train, Y_test, i)
            curr2, curr4 = calculate_accuracy(X_train.T[1:].T,
                                              X_test.T[1:].T,
                                              Y_train, Y_test, i)
            row_cross[1] += curr1
            row_cross[2] += curr2
            row_cross[3] += curr3
            row_cross[4] += curr4

        row_cross[1] /= n_splits
        row_cross[2] /= n_splits
        row_cross[3] /= n_splits
        row_cross[4] /= n_splits

        row_holdout = np.zeros(5)
        row_holdout[0] = i
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        row_holdout[1], row_holdout[3] = calculate_accuracy(X_train, X_test, Y_train, Y_test, i)
        row_holdout[2], row_holdout[4] = calculate_accuracy(X_train.T[1:].T, X_test.T[1:].T, Y_train, Y_test, i)

        table_cross = pd.concat([table_cross, pd.DataFrame(row_cross.reshape(-1, len(row_cross)), columns=columns)],
                                ignore_index=True)
        table_holdout = pd.concat(
            [table_holdout, pd.DataFrame(row_holdout.reshape(-1, len(row_cross)), columns=columns)], ignore_index=True)

    print_plot(table_cross.to_numpy(), table_holdout.to_numpy())


def calculate_accuracy(X_train, X_test, y_train, y_test, max_depth):
    decision_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42, criterion='gini')
    decision_tree.fit(X_train, y_train)

    # Calculate accuracy
    y_predict_test = decision_tree.predict(X_test)
    y_predict_train = decision_tree.predict(X_train)
    test_accuracy = np.sum(np.abs(np.array(y_predict_test) - np.array(y_test).flatten())) / len(y_predict_test)
    train_accuracy = np.sum(np.abs(np.array(y_predict_train) - np.array(y_train).flatten())) / len(y_predict_train)

    # Print accuracy
    # print(f"Test data accuracy: {1 - test_accuracy}")
    # print(f"Train data accuracy: {1 - train_accuracy}")

    return round(train_accuracy, 5), round(test_accuracy, 5)


def print_plot(table_cross, table_holdout):
    columns = ['Train error with id', 'Train error without id',
               'Test error with id', 'Test error without id']

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Cross validation vs Hold-out accuracy")

    ax1.plot(table_cross.T[0], table_cross.T[1])
    ax1.plot(table_cross.T[0], table_cross.T[2])
    ax1.plot(table_cross.T[0], table_cross.T[3])
    ax1.plot(table_cross.T[0], table_cross.T[4])
    ax1.legend(columns)
    ax1.set_title("Cross validation")
    ax1.set(xlabel="Decision Tree depth", ylabel="Error")

    ax2.plot(table_holdout.T[0], table_holdout.T[1])
    ax2.plot(table_holdout.T[0], table_holdout.T[2])
    ax2.plot(table_holdout.T[0], table_holdout.T[3])
    ax2.plot(table_holdout.T[0], table_holdout.T[4])
    ax2.legend(columns)
    ax2.set_title("Hold-out")
    ax2.set(xlabel="Decision Tree depth", ylabel="Error")

    fig.set_size_inches(13, 6)
    fig.show()

