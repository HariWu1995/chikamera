import os
import sys

import pandas as pd
import numpy as np

from ..core.trajdataset import TrajDataset


def load_chaos(path, separator, **kwargs):
    traj_dataset = TrajDataset()
    # TODO
    #  ChAOS Style: 1 file per agent, pos_x, pos_y
    print("\nLoad Chaos style: not implemented yet\n")
    return traj_dataset

