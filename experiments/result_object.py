from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Result:
    TFD_PATH = "TFD_PATH"
    TFD = "TFD"
    ARDA = "ARDA"
    JOIN_ALL = "TFD_JOIN_ALL"
    JOIN_ALL_FS = "TFD_JOIN_ALL_FS"
    BASE = "BASE"

    approach: str
    data_path: str
    data_label: str
    algorithm: str
    join_time: Optional[float] = None
    total_time: float = 0.0
    feature_selection_time: Optional[float] = None
    depth: Optional[int] = None
    accuracy: Optional[float] = None
    train_time: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    cutoff_threshold: Optional[float] = None
    redundancy_threshold: Optional[float] = None
    rank: Optional[int] = None

    def __post_init__(self):
        if self.join_time is not None:
            self.total_time += self.join_time

        if self.train_time is not None:
            self.total_time += self.train_time

        if self.feature_selection_time is not None:
            self.total_time += self.feature_selection_time
