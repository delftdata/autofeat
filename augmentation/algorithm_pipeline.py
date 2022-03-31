from timeit import default_timer as timer

import pandas as pd

from augmentation.algorithm_steps import select_join_paths, process_join_paths, rank_join_paths, \
    get_top_k_from_dict, train_augmented
from feat_sel import FeatSel
from algorithms import CART, ID3, XGB

ALL_FEAT_SEL = [FeatSel.SU, FeatSel.GAIN, FeatSel.GINI, FeatSel.CORR, FeatSel.RELIEF]
ALGORITHMS = [CART, ID3, XGB]

datasets = {
    "other-data/decision-trees-split/football/football.csv": ["win", "id"],
    "other-data/decision-trees-split/kidney-disease/kidney_disease.csv": ["classification", "id"],
    "other-data/decision-trees-split/steel-plate-fault/steel_plate_fault.csv": ["Class", "index"],
    "other-data/decision-trees-split/titanic/titanic.csv": ["Survived", "PassengerId"]
}


def pipeline(data, k=None):
    if k:
        filename = f'pipeline-{k}'
    else:
        filename = 'pipeline-all'

    results = []
    for path, features in data.items():
        label = features[0]
        ids = features[1]
        top_k = []
        start_enumerate = timer()
        join_paths = select_join_paths(path, label)
        processed = process_join_paths(join_paths)
        end_enumerate = timer()

        for feat_sel in ALL_FEAT_SEL:
            start_rank = timer()
            ranked = rank_join_paths(processed, label, feat_sel)
            if k and k > 0:
                top_k = get_top_k_from_dict(ranked, 1)
            end_rank = timer()

            for alg in ALGORITHMS:
                runtime = (end_enumerate - start_enumerate) + (end_rank - start_rank)
                if len(top_k) > 0:
                    res = train_augmented(top_k, processed, label, alg, runtime, feat_sel)
                else:
                    res = train_augmented(ranked, processed, label, alg, runtime, feat_sel)
                results = results + res

        print(results)

    df = pd.DataFrame(results)
    df.to_csv(f'../results/{filename}.csv', index=False)


# pipeline(datasets, k=1)  # BestRank
# pipeline(datasets)  # Top-k

