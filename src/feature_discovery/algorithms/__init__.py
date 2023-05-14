from typing import List

from feature_discovery.algorithms.base_algorithm import BaseAlgorithm
from feature_discovery.algorithms.cart import CART
from feature_discovery.algorithms.id3 import ID3
from feature_discovery.algorithms.xgb import XGB

TRAINING_FUNCTIONS: List[BaseAlgorithm] = [CART, XGB, ID3]
