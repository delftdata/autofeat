import ast
import os

import pandas as pd

from augmentation.train_algorithms import train_CART_and_print

non_aug_data = {
    "other-data/decision-trees-split/football/football.csv": ["win", "id"],
    "other-data/decision-trees-split/kidney-disease/kidney_disease.csv": ["classification", "id"],
    "other-data/decision-trees-split/steel-plate-fault/steel_plate_fault.csv": ["Class", "index"],
    "other-data/decision-trees-split/titanic/titanic.csv": ["Survived", "PassengerId"]
}

join_all_data = {
    # 'joined-df/join-all-football.csv': 'win',
    'joined-df/join-all-kidney_disease.csv': 'classification',
    # 'joined-df/join-all-steel_plate_fault.csv': 'Class',
    # 'joined-df/join-all-titanic.csv': 'Survived'
}


df_labels = {
    'football': 'win',
    'kidney_disease': 'classification',
    'steel_plate_fault': 'Class',
    'titanic': 'Survived'
}


folder_name = os.path.abspath(os.path.dirname(__file__))


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


def print_trees_baseline(data):
    results = []

    for path, features in data.items():
        dataset_name = path.split('.')[0].split('/')[-1]
        label = features[0]
        ids = features[1]
        X, y = read_data(path, label, [ids])

        accuracy, params = train_CART_and_print(X, y, dataset_name)
        res = {
            'dataset': path,
            'accuracy': accuracy,
        }
        res.update(params)
        results.append(res)

    print(results)
    return results


def print_trees_join_all(data):
    results = []

    for path, label in data.items():
        dataset_name = path.split('.')[0].split('-')[-1]

        dataset_filepath = os.path.join(folder_name, f"../{path}")
        dataframe = pd.read_csv(dataset_filepath, header=0, engine="python", encoding="utf8", quotechar='"', escapechar='\\')

        print("\tEncoding data")
        dataframe = dataframe.apply(lambda x: pd.factorize(x)[0])
        X = dataframe.drop(columns=[label])
        y = dataframe[label]

        accuracy, params = train_CART_and_print(X, y, dataset_name)
        res = {
            'dataset': path,
            'accuracy': accuracy,
        }
        res.update(params)
        results.append(res)

    print(results)
    return results


def shorten_dataset_name_join(data):
    split1 = data.split('.')[0]
    split2 = split1.split('/')[-1]
    split3 = split2.split('-')[0]
    return split3


def print_trees_best_path():
    dataset = 'results/pipeline-best.csv'
    dataset_filepath = os.path.join(folder_name, f"../{dataset}")
    dataframe = pd.read_csv(dataset_filepath, header=0, engine="python", encoding="utf8", quotechar='"',
                            escapechar='\\')
    dataframe['base_table'] = dataframe['dataset'].apply(lambda x: shorten_dataset_name_join(x))

    cart = dataframe[dataframe['algorithm'] == 'CART']
    cart_best = cart.groupby(['base_table', 'dataset', 'path_ids'])['accuracy'].max().reset_index()
    cart_best['path_ids'] = cart_best['path_ids'].apply(ast.literal_eval)

    results = []
    for i, row in cart_best.iterrows():
        label = df_labels[row['base_table']]
        df = pd.read_csv(os.path.join(folder_name, f"joined-df/{row['dataset']}"), header=0, engine="python", encoding="utf8", quotechar='"',
                            escapechar='\\')
        df.drop(columns=row['path_ids'], inplace=True)
        df = df.apply(lambda x: pd.factorize(x)[0])
        X = df.drop(columns=[label])
        y = df[label]
        accuracy, params = train_CART_and_print(X, y, row['dataset'])
        res = {
            'dataset': row['dataset'],
            'accuracy': accuracy,
        }
        res.update(params)
        results.append(res)

    print(results)
    return results


# res1 = print_trees_baseline(non_aug_data)
# df = pd.DataFrame(res1)
# df.to_csv('results/from-img/non_aug.csv', index=False)

# res2 = print_trees_join_all(join_all_data)
# df = pd.DataFrame(res2)
# df.to_csv('results/from-img/join_all.csv', index=False)
#
res3 = print_trees_best_path()
# df = pd.DataFrame(res3)
# df.to_csv('results/from-img/best.csv', index=False)
