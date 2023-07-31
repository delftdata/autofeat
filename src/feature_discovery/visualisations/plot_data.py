from json import loads
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from itertools import cycle
from math import pi

from feature_discovery.config import RESULTS_FOLDER, PLOTS_FOLDER


def parse_feature_importance(dataframe):
    result = dataframe['feature_importance']
    number_of_features = []
    features = []
    for i in result:
        j = i.replace("\'", "\"")
        js = loads(j)
        number_of_features.append(len(js.values()))
        features.append(list(js.keys()))

    return number_of_features, features


def parse_join_path_features(dataframe):
    dataframe['join_path_features'] = dataframe['join_path_features'].fillna('')
    result = dataframe['join_path_features']

    number_of_features = []
    jp_features = []
    for i in result:
        j = i.replace("[", "")
        j = j.replace("]", "")
        j = j.replace("'", "")
        k = j.split(", ")

        if len(k) == 1:
            number_of_features.append(0)
        else:
            number_of_features.append(len(k))
        jp_features.append(k)

    return number_of_features, jp_features


def determine_common_features(features, jp_features):
    nr_common_features = []
    difference = []

    for i, values in enumerate(features):
        set_a = set(values)
        set_b = set(jp_features[i])
        nr_common_features.append(len(set_a.intersection(set_b)))
        difference.append(set_b - set_a)

    return nr_common_features, difference


def parse_data(dataframe):
    number_of_features, features = parse_feature_importance(dataframe)
    dataframe['number_features_importance'] = number_of_features

    number_of_features, jp_features = parse_join_path_features(dataframe)
    dataframe['number_join_path_features'] = number_of_features

    nr_common_features, difference = determine_common_features(features, jp_features)
    dataframe['nr_common_features'] = nr_common_features
    dataframe['different_features'] = difference

    return dataframe

