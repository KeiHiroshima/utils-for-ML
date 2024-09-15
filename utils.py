import argparse
import os
import pickle
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def timer(name):
    """
    def wait(sec: float):
    time.sleep(sec)
    with timer("wait"):
        wait(2.0)

    """
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


class Logger:
    def __init__(self, log_list=[]):
        self.log = {}
        for n in log_list:
            self.log[n] = []

    def set(self, name, value):
        self.log[name].append(value)

    def get(self, name):
        return self.log[name]

    def save(self, savedir="log", name="log.pkl"):
        path = os.path.join(savedir, name)
        with open(path, "wb") as f:
            pickle.dump(self.log, f)

    def load(self, savedir="log", name="log.pkl"):
        path = os.path.join(savedir, name)
        with open(path, "rb") as f:
            self.log = pickle.load(f)

        return self.log


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def set_random_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
