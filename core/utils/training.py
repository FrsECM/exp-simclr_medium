import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import torch

def Get_Lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']