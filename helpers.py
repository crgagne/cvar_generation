import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

def set_seeds(seed = 2311):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
