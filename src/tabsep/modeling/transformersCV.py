import os

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

CORES_AVAILABLE = len(os.sched_getaffinity(0))
torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
