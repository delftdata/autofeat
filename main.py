from data_ingestion import ingest_data


def main():
    path = "other-data/auto-fabricated/titanic/correlation"
    ingest_data.ingest_fabricated_data(path)
    ingest_data.ingest_connections(path)


if __name__ == '__main__':
    main()