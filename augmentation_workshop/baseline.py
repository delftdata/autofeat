import os

import pandas as pd

from augmentation.train_algorithms import train_CART, train_ID3, train_XGBoost

datasets = {
    "other-data/decision-trees-split/football/football.csv": ["win", "id"],
    "other-data/decision-trees-split/kidney-disease/kidney_disease.csv": ["classification", "id"],
    "other-data/decision-trees-split/steel-plate-fault/steel_plate_fault.csv": ["Class", "index"],
    "other-data/decision-trees-split/titanic/titanic.csv": ["Survived", "PassengerId"]
}

folder_name = os.path.abspath(os.path.dirname(__file__))
algorithms = ['CART', 'ID3', 'XGBoost']


def read_data(filename: str, label: str, ids: list):
    base_filepath = os.path.join(folder_name, f"../{filename}")
    aug_df = pd.read_csv(base_filepath, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')

    columns_to_drop = list(set(aug_df.columns) & set(ids))
    print(f'Dropping columns: {columns_to_drop}')
    dataset = aug_df.drop(columns=columns_to_drop)

    print("\tEncoding data")
    df = dataset.apply(lambda x: pd.factorize(x)[0])

    print("\tSplit X, y")
    y = df[label]
    X = df.copy().drop([label], axis=1)

    print(f'Shape: \t {X.shape}')
    print(X.columns)

    return X, y


def baseline(data):
    results = []

    for path, features in data.items():
        label = features[0]
        ids = features[1]
        X, y = read_data(path, label, [ids])

        accuracy, params = train_CART(X, y)
        res = {
            'dataset': path,
            'accuracy': accuracy,
            'algorithm': 'CART'
        }
        res.update(params)
        results.append(res)

        accuracy = train_ID3(X, y)
        res = {
            'dataset': path,
            'accuracy': accuracy,
            'algorithm': 'ID3'
        }
        results.append(res)

        accuracy, params = train_XGBoost(X, y)
        res = {
            'dataset': path,
            'accuracy': accuracy,
            'algorithm': 'XGBoost'
        }
        res.update(params)
        results.append(res)

    print(results)

    df = pd.DataFrame(results)
    df.to_csv('../results/auto-fabricated/baseline.csv', index=False)


data_path = {
    "other-data/auto-fabricated/titanic/random_overlap/table_0_0.csv": ["Survived", "PassengerId"],
}
baseline(data_path)
