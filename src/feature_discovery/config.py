import os
from pathlib import Path

ROOT_FOLDER = Path(
    os.getenv("TFD_ROOT_FOLDER", Path(os.path.abspath(__file__)).parent.parent.parent.resolve())
).resolve()

CONNECTIONS = "connections.csv"

DATASET_TYPE = "benchmark"

DATA = "data"
DATA_FOLDER = ROOT_FOLDER / DATA / DATASET_TYPE
RESULTS_FOLDER = ROOT_FOLDER / "results" / "revision-test"
# RESULTS_FOLDER = ROOT_FOLDER / "results"
AUTO_GLUON_FOLDER = ROOT_FOLDER / "AutogluonModels"

### CREDENTIALS ###
# NEO4J_HOST = os.getenv("NEO4J_HOST", "bolt://localhost:11003")
NEO4J_HOST = os.getenv("NEO4J_HOST", "neo4j://172.30.106.77:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASS = os.getenv("NEO4J_PASS", "")
NEO4J_CREDENTIALS = (NEO4J_USER, NEO4J_PASS)

NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", DATASET_TYPE)
# NEO4J_DATABASE = "lake"
