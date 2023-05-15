import os
from pathlib import Path

ROOT_FOLDER = Path(
    os.getenv("TFD_ROOT_FOLDER", Path(os.path.abspath(__file__)).parent.parent.parent.resolve())
).resolve()

# MAPPING = "mapping.json"
MAPPING = "mapping_"
JSON = ".json"
CONNECTIONS = "connections.csv"
VALENTINE_CONNECTIONS = "valentine-connections.csv"
ENUMERATED_PATHS = "enumerated-paths.json"
TRAINING_DATASET = "results.csv"
JOINED_PATHS = "joined-paths.json"
RANKING_FUNCTION = 'ranking-func.json'
RANKING_VERIFY = 'verify_ranking.csv'
ALL_PATHS = 'all_paths'

DATA = "data"
MAPPING_FOLDER = ROOT_FOLDER / 'mappings' / '2024'
JOIN_RESULT_FOLDER = ROOT_FOLDER / 'joined-df'
DATA_FOLDER = ROOT_FOLDER / DATA
PLOTS_FOLDER = ROOT_FOLDER / "plots" / "2024"
RESULTS_FOLDER = ROOT_FOLDER / "results"

ACCURACY_RESULTS_ALL_PNG = PLOTS_FOLDER / 'accuracy-results-all.png'

### CREDENTIALS ###
NEO4J_HOST = os.getenv("NEO4J_HOST", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "pass1234")
NEO4J_CREDENTIALS = (NEO4J_USER, NEO4J_PASS)
