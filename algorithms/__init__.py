from typing import List

from algorithms.base_algorithm import BaseAlgorithm
from algorithms.cart import CART
from algorithms.id3 import ID3
from algorithms.xgb import XGB

TRAINING_FUNCTIONS: List[BaseAlgorithm] = [CART, XGB, ID3]
