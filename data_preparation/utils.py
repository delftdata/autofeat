import json
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from utils.file_naming_convention import MAPPING_FOLDER, ENUMERATED_PATHS

folder_name = os.path.abspath(os.path.dirname(__file__))


def get_paths():
    with open(f"{os.path.join(folder_name, '../', MAPPING_FOLDER)}/{ENUMERATED_PATHS}", 'r') as fp:
        all_paths = json.load(fp)
    return all_paths


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


def _get_join_path(path: str):
    # Path looks like table_source/table_name/key--table_source...
    # Join path looks like table_source--table_name--table_source...

    path_tokens = path.split("--")
    join_path_tokens = ["--".join(token.split('/')[:-1]) for token in path_tokens]
    return "--".join(join_path_tokens)

