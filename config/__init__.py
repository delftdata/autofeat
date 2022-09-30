from pathlib import Path

ROOT_FOLDER = Path.cwd().parents[0].resolve()

MAPPING = "mapping.json"
CONNECTIONS = "connections.csv"
ENUMERATED_PATHS = "enumerated-paths.json"
TRAINING_DATASET = "results.csv"
JOINED_PATHS = "joined-paths.json"
RANKING_FUNCTION = 'ranking-func.json'
RANKING_VERIFY = 'verify_ranking.csv'
ALL_PATHS = 'all_paths'

MAPPING_FOLDER = ROOT_FOLDER / 'mappings' / 'revision'
JOIN_RESULT_FOLDER = ROOT_FOLDER / 'joined-df' / 'revision'
DATA_FOLDER = ROOT_FOLDER / 'other-data' / 'synthetic'
PLOTS_FOLDER = ROOT_FOLDER / "plots"
