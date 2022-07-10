import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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
    min_value = min(dictionary.values())
    if min_value < 0:
        aux_dict = {k: v - min_value for k, v in aux_dict.items()}

    max_value = max(aux_dict.values())
    return {key: value/max_value for key, value in aux_dict.items()}


def prepare_data_for_ml(dataframe, target_column):
    df = dataframe.fillna(0)
    df = df.apply(lambda x: pd.factorize(x)[0] if x.dtype == object else x)
    X = df.drop(columns=[target_column])

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    normalized_X = pd.DataFrame(scaled_X, columns=X.columns)
    # print(normalized_X)

    y = df[target_column].astype(int)

    return normalized_X, y

