from subprocess import check_call
import numpy as np
import time
import six
import sys
# This is for Id3Estimator, see: https://stackoverflow.com/questions/61867945/python-import-error-cannot-import-name-six-from-sklearn-externals
sys.modules['sklearn.externals.six'] = six

from id3 import Id3Estimator
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from xgboost import XGBClassifier

from utils.util_functions import get_ID3_tree_depth




num_cv = 10


def train_CART(X, y, do_sfs: bool = False):
    sfs_time = None

    if do_sfs:
        start = time.time()
        decision_tree = tree.DecisionTreeClassifier()
        sfs = SFS(estimator=decision_tree, forward=True, k_features="best", n_jobs=1, cv=5)
        sfs.fit(X, y)
        X = sfs.transform(X)
        end = time.time()
        sfs_time = end - start
        print("==== Finished SFS =====")

    print("Split data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    print(f"X train {X_train.shape}, X_test {X_test.shape}")
    print(f"Y uniqueness: {len(y_train.dropna()) / len(y_train)}")

    print("\tFinding best tree params")

    parameters = {"criterion": ["entropy", "gini"], "max_depth": range(1, X_train.shape[1] + 1)}

    print(parameters)

    decision_tree = tree.DecisionTreeClassifier()
    grids = GridSearchCV(decision_tree, parameters, n_jobs=1, scoring="accuracy", cv=15)
    grids.fit(X_train, y_train)
    params = grids.best_params_

    # TODO Store all the accuracies and the trees and see how the trees look
    print(f"Hyper-params: {params} for best score: {grids.best_score_}")

    print(f"\t Training ... ")

    decision_tree = grids.best_estimator_
    start = time.time()
    cv_output = cross_validate(
        estimator=decision_tree,
        X=X,
        y=y,
        scoring="accuracy",
        return_estimator=True,
        cv=num_cv,
        verbose=10,
        n_jobs=1,
    )
    end = time.time()
    train_time = end - start
    feature_importances = [estimator.feature_importances_ for estimator in cv_output["estimator"]]

    acc_decision_tree = np.mean(cv_output["test_score"])
    feature_importance = np.around(np.median(feature_importances, axis=0), 3)

    # y_pred = decision_tree.predict(X_test)
    # acc_decision_tree = round(accuracy_score(y_test, y_pred) * 100, 2)
    print(f"\t\tAccuracy CART: {acc_decision_tree}")

    return acc_decision_tree, params, feature_importance, train_time, sfs_time


def train_CART_and_print(X, y, dataset_name, plot_path):
    print("Split data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    print(f"X train {X_train.shape}, X_test {X_test.shape}")
    print(f"Y uniqueness: {len(y_train.dropna()) / len(y_train)}")

    print("\tFinding best tree params")
    parameters = {"criterion": ["entropy", "gini"], "max_depth": range(1, len(list(X_train)) + 1)}
    decision_tree = tree.DecisionTreeClassifier()
    grids = GridSearchCV(decision_tree, parameters, n_jobs=4, scoring="accuracy", cv=15)
    grids.fit(X_train, y_train)
    params = grids.best_params_
    print(f"Hyper-params: {params} for best score: {grids.best_score_}")

    print(f"\t Training ... ")
    decision_tree = tree.DecisionTreeClassifier(
        max_depth=params["max_depth"], criterion=params["criterion"]
    )
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(accuracy_score(y_test, y_pred) * 100, 2)
    print(f"\t\tAccuracy CART: {acc_decision_tree}")

    with open(f"{plot_path}/tree.dot", "w") as f:
        tree.export_graphviz(
            decision_tree,
            out_file=f,
            max_depth=params["max_depth"],
            impurity=True,
            feature_names=list(X_train),
            class_names=list(map(lambda x: str(x), y_train.unique())),
            rounded=True,
            filled=True,
        )

    # Convert .dot to .png to allow display in web notebook
    check_call(["dot", "-Tpng", f"{plot_path}/tree.dot", "-o", f"{plot_path}/{dataset_name}.png"])

    # tree.plot_tree(decision_tree, fontsize=10)
    # plt.savefig(f'tree-img/tree-{dataset_name}.png', dpi=300, bbox_inches="tight")
    #
    # text = tree.export_text(decision_tree)
    # print(text)

    # Annotating chart with PIL
    # img = Image.open("tree1.png")
    # draw = ImageDraw.Draw(img)
    # # font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
    # draw.text((10, 0),  # Drawing offset (position)
    #           f'{title}',  # Text to draw
    #           (0, 0, 255)  # RGB desired color
    #           )  # ImageFont object with desired font
    # img.save(f'sample-out-{label_col}.png')
    # PImage("sample-out.png")

    return acc_decision_tree, params, decision_tree.feature_importances_


def train_ID3(X, y, do_sfs: bool = False):
    sfs_time = None

    # Not supported yet. TODO: Adapt ID3
    if do_sfs:
        start = time.time()
        decision_tree = Id3Estimator()
        sfs = SequentialFeatureSelector(
            estimator=decision_tree, n_features_to_select="auto", scoring="accuracy"
        )
        sfs.fit(X, y)
        X = sfs.transform(X)
        end = time.time()
        sfs_time = end - start

    decision_tree = Id3Estimator()
    start = time.time()
    cv_output = cross_validate(
        estimator=decision_tree,
        X=X,
        y=y,
        scoring="accuracy",
        return_estimator=True,
        verbose=10,
        cv=num_cv,
        n_jobs=1,
    )
    end = time.time()
    train_time = end - start
    max_depths = [get_ID3_tree_depth(estimator.tree_.root) for estimator in cv_output["estimator"]]
    params = {"max_depth": np.median(max_depths)}

    acc_decision_tree = np.mean(cv_output["test_score"])

    print(f"\t\tAccuracy ID3: {acc_decision_tree}")
    print(max_depths)

    # Empty list to be consistent with other models
    return acc_decision_tree, params, [], train_time, sfs_time


def train_XGBoost(X, y, do_sfs: bool = False):
    sfs_time = None

    if do_sfs:
        start = time.time()
        decision_tree = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            # Reduce execution time, otherwise it explodes
            max_depth=5,
            n_jobs=1,
        )
        sfs = SFS(
            estimator=decision_tree, forward=True, k_features="best", n_jobs=-1, cv=5, verbose=3
        )
        sfs.fit(X, y)
        X = sfs.transform(X)
        end = time.time()
        sfs_time = end - start
        print("==== Finished SFS =====")

    print("Split data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    print(f"X train {X_train.shape}, X_test {X_test.shape}")
    print(f"Y uniqueness: {len(y_train.dropna()) / len(y_train)}")

    parameters = {"max_depth": range(1, X_train.shape[1] + 1)}
    print(parameters)
    decision_tree = XGBClassifier(
        objective="binary:logistic", eval_metric="auc", use_label_encoder=False, n_jobs=1
    )
    grids = GridSearchCV(decision_tree, parameters, n_jobs=1, scoring="accuracy", cv=10)
    grids.fit(X_train, y_train)
    params = grids.best_params_
    print(f"Hyper-params: {params} for best score: {grids.best_score_}")

    decision_tree = grids.best_estimator_

    start = time.time()
    cv_output = cross_validate(
        estimator=decision_tree,
        X=X,
        y=y,
        scoring="accuracy",
        return_estimator=True,
        cv=num_cv,
        verbose=10,
    )
    end = time.time()
    feature_importances = [estimator.feature_importances_ for estimator in cv_output["estimator"]]
    acc_decision_tree = np.mean(cv_output["test_score"])
    feature_importance = np.around(np.median(feature_importances, axis=0), 3)
    train_time = end - start

    print(f"\t\tAccuracy XGBoost: {acc_decision_tree}")

    return acc_decision_tree, params, feature_importance, train_time, sfs_time


def apply_cross_validation(X, y, tree, n_splits=10):
    scores = cross_val_score(estimator=tree, X=X, y=y, cv=n_splits)
    return scores.mean()
