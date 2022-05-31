import torch
import numpy as np


def straight_line_td(data_1, data_2, num_of_points):
    df = (data_2 - data_1) / (num_of_points - 1)
    line_data = [data_1 + i * df for i in range(num_of_points)]
    return line_data
