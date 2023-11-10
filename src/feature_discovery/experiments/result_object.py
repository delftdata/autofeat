from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class Result:
    TFD_PATH = "TFD_PATH"
    TFD = "AutoFeat"
    TFD_REL = "AutoFeat_Rel"
    TFD_RED = "AutoFeat_Red"
    TFD_Pearson = "AutoFeat-Pearson-MRMR"
    TFD_Pearson_JMI = "AutoFeat-Pearson-JMI"
    TFD_JMI = "AutoFeat-Spearman-JMI"
    ARDA = "ARDA"
    JOIN_ALL_BFS = "Join_All_BFS"
    JOIN_ALL_BFS_F = "Join_All_BFS_Filter"
    JOIN_ALL_BFS_W = "Join_All_BFS_Wrapper"
    JOIN_ALL_DFS = "Join_All_DFS"
    JOIN_ALL_DFS_F = "Join_All_DFS_Filter"
    JOIN_ALL_DFS_W = "Join_All_DFS_Wrapper"
    BASE = "BASE"

    algorithm: str
    data_path: str = None
    approach: str = None
    data_label: str = None
    join_time: Optional[float] = None
    total_time: float = 0.0
    feature_selection_time: Optional[float] = None
    depth: Optional[int] = None
    accuracy: Optional[float] = None
    train_time: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    join_path_features: List[str] = None
    cutoff_threshold: Optional[float] = None
    redundancy_threshold: Optional[float] = None
    rank: Optional[int] = None
    top_k: int = None

    def __post_init__(self):
        if self.join_time is not None:
            self.total_time += self.join_time

        if self.train_time is not None:
            self.total_time += self.train_time

        if self.feature_selection_time is not None:
            self.total_time += self.feature_selection_time
