import json
import os

from feature_discovery.config import (
    MAPPING_FOLDER,
    ENUMERATED_PATHS,
)

folder_name = os.path.abspath(os.path.dirname(__file__))


def get_paths():
    with open(MAPPING_FOLDER / ENUMERATED_PATHS, "r") as fp:
        all_paths = json.load(fp)
    return all_paths


def get_join_path(path: str) -> str:
    # Path looks like table_source/table_name/key--table_source...
    # Join path looks like table_source--table_name--table_source...

    path_tokens = path.split("--")
    join_path_tokens = ["--".join(token.split("/")[:-1]) for token in path_tokens]
    return "--".join(join_path_tokens)


def get_path_length(path: str) -> int:
    # Path looks like table_source/table_name/key--table_source...
    path_tokens = path.split("--")
    # Length = 1 means that we have 2 tables
    return len(path_tokens) - 1


def compute_join_name(join_key_property: tuple, partial_join_name: str) -> str:
    """
    Compute the name of the partial join, given the properties of the new join and the previous join name.

    :param join_key_property: (neo4j relation property, outbound label, inbound label)
    :param partial_join_name: Name of the partial join.
    :return: The name of the next partial join
    """
    join_prop, from_table, to_table = join_key_property
    joined_path = f"{partial_join_name}--{from_table}-{join_prop['from_column']}-{join_prop['to_column']}-{to_table}"
    return joined_path
