from plot_experiments import read_data, plot
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


def plot_titanic():
    train_csv = "titanic_train.csv"
    y_column = 'Survived'

    X, y = read_data(train_csv, y_column)

    si = SimpleImputer(strategy='most_frequent')
    X[['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']] = si.fit_transform(
        X[['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']])

    si = SimpleImputer(strategy='mean')
    X[['Age', 'Fare']] = si.fit_transform(X[['Age', 'Fare']])

    X = X.drop(['Name', 'Ticket'], axis=1)

    si = SimpleImputer(strategy='constant', fill_value="-1")
    X[['Cabin']] = si.fit_transform(X[['Cabin']])

    enc = OrdinalEncoder()
    X[['Sex', 'Embarked', 'Cabin']] = enc.fit_transform(
        X[['Sex', 'Embarked', 'Cabin']])

    X = X.reset_index().drop(['index', 'PassengerId'], axis=1)
    X[['id']] = X.index

    cols = X.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    X = X[cols]

    plot(X.to_numpy(), y.to_numpy())


if __name__ == '__main__':
    plot_titanic()
