import json
import os

from experiments.verify_ranking import verify_join_no_pruning


folder_name = os.path.abspath(os.path.dirname(__file__))
join_result_path = 'joined-df/wstitanic'
join_path = f"{folder_name}/{join_result_path}"
target_column = "Survived"
base_table = "titanic.csv"
# path = "other-data/auto-fabricated/titanic/random_overlap"
path = "other-data/decision-trees-split/titanic"
base_table_path = f"{os.path.join(folder_name, path, base_table)}"
mappings_path = "mappings/wstitanic"


def test_verify_join_no_pruning():
    with open(f"{os.path.join(folder_name, '../', mappings_path)}/ranking.json", 'r') as fp:
        sorted_ranking = json.load(fp)

    with open(f"{os.path.join(folder_name, '../', mappings_path)}/mapping.json", 'r') as fp:
        mapping = json.load(fp)

    result = verify_join_no_pruning(sorted_ranking, mapping, join_result_path, base_table, target_column)
    print(result)


if __name__ == '__main__':
    test_verify_join_no_pruning()
