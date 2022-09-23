import statistics

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def get_top_k_from_dict(join_paths: dict, k: int):
    return {key: join_paths[key] for i, key in enumerate(join_paths) if i < k}


def get_elements_less_than_value(dictionary: dict, value: float):
    return {k: v for k, v in dictionary.items() if abs(v) < value}


def get_elements_higher_than_value(dictionary: dict, value: float):
    return {k: v for k, v in dictionary.items() if v > value}


def normalize_dict_values(dictionary: dict):
    if len(dictionary.values()) < 2:
        return dictionary

    aux_dict = dictionary.copy()
    min_neg_value = min(dictionary.values())
    if min_neg_value < 0:
        aux_dict = {k: v - min_neg_value for k, v in aux_dict.items()}
    avg = statistics.mean(aux_dict.values())
    max_value = max(aux_dict.values())
    min_value = min(aux_dict.values())
    return {
        key: (value - avg) / (max_value - min_value) if (value - avg) != 0 else 0
        for key, value in aux_dict.items()
    }


def prepare_data_for_ml(dataframe, target_column):
    df = dataframe.fillna(0)
    df = df.apply(lambda x: pd.factorize(x)[0] if x.dtype == object else x)
    X = df.drop(columns=[target_column])

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    normalized_X = pd.DataFrame(scaled_X, columns=X.columns)
    # print(normalized_X)

    y = df[target_column].astype(int)

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name=target_column)

    return normalized_X, y


def transform_node_to_dict(node):
    dict_node = {'id': node.get('id'), 'label': node.get('label'), 'name': node.get('name'),
                 'source_name': node.get('source_name'), 'source_path': node.get('source_path')}
    return dict_node


