from dataclasses import dataclass
from typing import Dict, List, Set

from experiments.result_object import Result


@dataclass
class BfsDataModel:
    model_name: str
    # Store the accuracy from CART for each join path
    ranked_paths: Dict[str, Result] = None
    # Mapping with the name of the join and the corresponding name of the file containing the join result.
    join_name_mapping: Dict[str, str] = None
    # Save the selected features of the previous join path (used for conditional redundancy)
    partial_join_selected_features: Dict[str, List] = None
    current_queue: set = None
