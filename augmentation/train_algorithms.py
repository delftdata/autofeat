from subprocess import check_call

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

from augmentation.id3_alg import GadId3Classifier


def train_CART(X, y):
    print('Split data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    print(f'X train {X_train.shape}, X_test {X_test.shape}')
    print(f'Y uniqueness: {len(y_train.dropna()) / len(y_train)}')

    print('\tFinding best tree params')
    parameters = {'criterion': ['entropy', 'gini'], 'max_depth': range(1, len(list(X_train)) + 1)}
    decision_tree = tree.DecisionTreeClassifier()
    grids = GridSearchCV(decision_tree, parameters, n_jobs=4, scoring='accuracy', cv=15)
    grids.fit(X_train, y_train)
    params = grids.best_params_

    # TODO Store all the accuracies and the trees and see how the trees look
    print(f'Hyper-params: {params} for best score: {grids.best_score_}')

    print(f'\t Training ... ')
    decision_tree = tree.DecisionTreeClassifier(max_depth=params['max_depth'], criterion=params['criterion'],
                                                random_state=24)
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(accuracy_score(y_test, y_pred) * 100, 2)
    print(f'\t\tAccuracy CART: {acc_decision_tree}')

    return acc_decision_tree, params


def train_CART_and_print(X, y, dataset_name, path):
    print('Split data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    print(f'X train {X_train.shape}, X_test {X_test.shape}')
    print(f'Y uniqueness: {len(y_train.dropna()) / len(y_train)}')

    print('\tFinding best tree params')
    parameters = {'criterion': ['entropy', 'gini'], 'max_depth': range(1, len(list(X_train)) + 1)}
    decision_tree = tree.DecisionTreeClassifier()
    grids = GridSearchCV(decision_tree, parameters, n_jobs=4, scoring='accuracy', cv=15)
    grids.fit(X_train, y_train)
    params = grids.best_params_
    print(f'Hyper-params: {params} for best score: {grids.best_score_}')

    print(f'\t Training ... ')
    decision_tree = tree.DecisionTreeClassifier(max_depth=params['max_depth'], criterion=params['criterion'])
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(accuracy_score(y_test, y_pred) * 100, 2)
    print(f'\t\tAccuracy CART: {acc_decision_tree}')

    with open(f"{path}/tree.dot", 'w') as f:
        tree.export_graphviz(decision_tree,
                             out_file=f,
                             max_depth=params['max_depth'],
                             impurity=True,
                             feature_names=list(X_train),
                             class_names=list(map(lambda x: str(x), y_train.unique())),
                             rounded=True,
                             filled=True)

    # Convert .dot to .png to allow display in web notebook
    check_call(['dot', '-Tpng', f'{path}/tree.dot', '-o', f'{path}/{dataset_name}.png'])

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


def train_ID3(X, y):
    print('Split data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    print(f'X train {X_train.shape}, X_test {X_test.shape}')
    print(f'Y uniqueness: {len(y_train.dropna()) / len(y_train)}')

    decision_tree = GadId3Classifier()
    decision_tree.fit(X_train, y_train)
    # print(decision_tree.tree)
    y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(accuracy_score(y_test, y_pred) * 100, 2)
    print(f'\t\tAccuracy ID3: {acc_decision_tree}')

    return acc_decision_tree


def train_XGBoost(X, y):
    print('Split data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    print(f'X train {X_train.shape}, X_test {X_test.shape}')
    print(f'Y uniqueness: {len(y_train.dropna()) / len(y_train)}')

    parameters = {'max_depth': range(1, len(list(X_train)) + 1)}
    decision_tree = XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False)
    grids = GridSearchCV(decision_tree, parameters, n_jobs=4, scoring='accuracy', cv=10)
    grids.fit(X_train, y_train)
    params = grids.best_params_
    print(f'Hyper-params: {params} for best score: {grids.best_score_}')

    decision_tree = XGBClassifier(objective='binary:logistic', eval_metric='auc',
                                  max_depth=params['max_depth'], use_label_encoder=False)
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(accuracy_score(y_test, y_pred) * 100, 2)
    print(f'\t\tAccuracy XGBoost: {acc_decision_tree}')

    return acc_decision_tree, params
