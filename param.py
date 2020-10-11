import os
import numpy as np
import torch
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')
data_folder = "./data/"
data_folder_out = "./models/"


def save_path(num_model=None):
    if num_model is None:
        now = datetime.now()
        folder_str = now.strftime("%Y_%m_%d__%H_%M_%S")
    else:
        folder_str = str(num_model)

    folder_path_models = os.path.join(data_folder_out, folder_str)
    return folder_path_models


class Param:
    def __init__(self):
        self.k_array_len = 40
        self.k_min = 6
        self.k_max = 500

        self.k_array = None
        self.compute_karray()

        self.area_preservation = False
        self.fac_spec = 0.25
        self.fac_norm = 0.04

        self.filter_scale = 5e4

        self.status_log = True

    def compute_karray(self):
        self.k_array = [int(n) for n in
                        np.exp(np.linspace(np.log(self.k_min), np.log(self.k_max), self.k_array_len)).tolist()]

    def reset_karray(self):
        param_base = Param()
        self.k_array_len = param_base.k_array_len
        self.k_min = param_base.k_min
        self.k_max = param_base.k_max
        self.compute_karray()

    def from_dict(self, d):
        for key in d:
            if hasattr(self, key):
                self.__setattr__(key, d[key])

    def print_self(self):
        print("parameters: ")
        p_d = self.__dict__
        for k in p_d:
            print(k, ": ", p_d[k], "  ", end='')
        print("")


class DiffParam(Param):
    def __init__(self):
        super().__init__()
        self.sigma_sink = 0.1
        self.num_sink = 10


class DeepParam(DiffParam):
    def __init__(self):
        super().__init__()

        self.status_log = False

        self.k_array_len = 8
        self.k_min = 6
        self.k_max = 21

        self.compute_karray()

        self.lr = 1e-4
        self.log_freq = 2
        self.batch_size = 10

        self.subsample_num = 2000
        self.num_hidden_layers = 32

        self.fac_spec = 0.3
        self.fac_norm = 0.025
        self.num_sink = 3
        self.num_hidden_layers = 120
        self.batch_size = 53
