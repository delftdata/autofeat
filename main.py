from augmentation import pipeline
from augmentation.pipeline import join_tables, join_tables_recursive
from data_ingestion import ingest_data


def main():
    path = "other-data/auto-fabricated/titanic/random_overlap"
    mapping = ingest_data.ingest_fabricated_data(path)
    # ingest_data.ingest_connections(path, mapping)
    # ingest_data.profile_valentine_all(path, mapping)
    all_paths = pipeline.path_enumeration()
    base_table = "table_0_0.csv"
    # joined_mapping = join_tables(all_paths, mapping, "table_0_0.csv")
    allp = []
    join_tables_recursive(all_paths, mapping, base_table, "", allp)
    # print(allp)


def test_path_enumeration():
    pipeline.path_enumeration()


if __name__ == '__main__':
    main()
    # test_path_enumeration()
