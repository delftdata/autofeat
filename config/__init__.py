import os
from pathlib import Path

ROOT_FOLDER = Path(os.path.abspath(__file__)).parent.parent.resolve()

# MAPPING = "mapping.json"
MAPPING = "mapping-cs.json"
CONNECTIONS = "connections.csv"
VALENTINE_CONNECTIONS = "valentine-connections.csv"
ENUMERATED_PATHS = "enumerated-paths.json"
TRAINING_DATASET = "results.csv"
JOINED_PATHS = "joined-paths.json"
RANKING_FUNCTION = 'ranking-func.json'
RANKING_VERIFY = 'verify_ranking.csv'
ALL_PATHS = 'all_paths'

MAPPING_FOLDER = ROOT_FOLDER / 'mappings' / '2024' / "tables"
JOIN_RESULT_FOLDER = ROOT_FOLDER / 'joined-df' / '2024' / "tables"
# DATA_FOLDER = ROOT_FOLDER / Path("..") / 'data' / 'nyc'
DATA_FOLDER = ROOT_FOLDER / 'data' / 'cs'
# DATA_FOLDER = ROOT_FOLDER / 'data' / 'ARDA'
# DATA_FOLDER = ROOT_FOLDER / 'other-data' / 'synthetic'
PLOTS_FOLDER = ROOT_FOLDER / "plots"
RESULTS_FOLDER = ROOT_FOLDER / "results"

ACCURACY_RESULTS_ALL_PNG = PLOTS_FOLDER / 'accuracy-results-all.png'

VALENTINE_THRESHOLD = 0.8
