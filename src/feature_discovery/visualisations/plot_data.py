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


def plot_accuracy(dataframe):
    accuracy = dataframe.groupby(['data_label', 'approach', 'algorithm'])[['accuracy']].max().reset_index()

    sns.set(font_scale=1.5, style="whitegrid")
    # plt.subplots(figsize=(20, 4))

    g = sns.catplot(x="data_label", y="accuracy", hue="approach", col="algorithm",
                    data=accuracy, kind="bar", height=4, aspect=1)

    g.set_titles("{col_name} accuracy")
    g.set_xlabels('Dataset')
    g.set_ylabels('Accuracy')

    loc, labels = plt.xticks()
    hh, ll = plt.gca().get_legend_handles_labels()

    g.set_xticklabels(labels, rotation=30)
    g._legend.remove()

    plt.legend(hh, ll, bbox_to_anchor=(-1, -0.85), loc=4, ncol=3, title="Approach")

    g.savefig(PLOTS_FOLDER / 'accuracy.png', dpi=300, bbox_inches="tight")


def plot_time(dataframe):
    total_time = dataframe.groupby(['data_label', 'approach', 'algorithm'])[
        ['accuracy', 'total_time']].max().reset_index()

    total_time['total_time'] = total_time['total_time'].apply(np.log10)

    sns.set(font_scale=1.5, style="whitegrid")
    # plt.subplots(figsize=(20, 4))

    g = sns.catplot(x="data_label", y="total_time", hue="approach", col="algorithm",
                    data=total_time, kind="bar", height=4, aspect=1)

    g.set_titles("{col_name} total_time")
    g.set_xlabels('Dataset')
    g.set_ylabels('Total time (log scale)')

    loc, labels = plt.xticks()
    hh, ll = plt.gca().get_legend_handles_labels()

    g.set_xticklabels(labels, rotation=30)
    g._legend.remove()

    plt.legend(hh, ll, bbox_to_anchor=(-1, -0.85), loc=4, ncol=3, title="Approach")

    g.savefig(PLOTS_FOLDER / 'time.png', dpi=300, bbox_inches="tight")


