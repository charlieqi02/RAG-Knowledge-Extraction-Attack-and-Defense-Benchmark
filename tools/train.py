"""Training utils."""
import datetime
import os
import random

import numpy as np
import torch



def get_savedir(args):
    """Get unique saving directory name."""
    if not args.continue_pipe:
        if not args.debug:
            dt = datetime.datetime.now()
            date = dt.strftime("%m_%d")
            save_dir = os.path.join(
                os.environ["LOG_DIR"], date, f"{args.rag}-{args.dataset}",
                f"{args.attack}-{args.defense}" + dt.strftime('-%H_%M_%S')
            )
            os.makedirs(save_dir)
        else:
            save_dir = os.path.join(os.environ["LOG_DIR"], "debug", f"{args.rag}-{args.attack}-{args.defense}")
            os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = args.continue_dir
    return save_dir


def set_seed(seed):
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)  # PyTorch random number generator.
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python software environment.
    
    if torch.cuda.is_available():  # CUDA random number generator.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # To ensure that CUDA convolution is deterministic.
        torch.backends.cudnn.deterministic = True  
        # If true, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
        torch.backends.cudnn.benchmark = False  