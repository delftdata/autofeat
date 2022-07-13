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

    plot(X, y, 'PassengerId')


if __name__ == '__main__':
    plot_titanic()
