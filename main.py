from data_ingestion import ingest_data


def main():
    path = "other-data/auto-fabricated/titanic/random_overlap"
    mapping = ingest_data.ingest_fabricated_data(path)
    ingest_data.ingest_connections(path, mapping)
    ingest_data.profile_valentine_all(path, mapping)


if __name__ == '__main__':
    main()
