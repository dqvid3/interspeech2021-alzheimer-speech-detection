import random
import numpy as np
import torch
import os

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
        # These settings may slow down training, but are
        # necessary for full reproducibility on cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set an environment variable for reproducibility of certain operations
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    print(f"Seed set to: {seed_value}")