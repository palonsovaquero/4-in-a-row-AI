import numpy as np
from networks import DQN, update_weights
from utils import play_a_game, play_against_computer
import torch
import torch.nn as nn

players = ["computer", "human"]

models = [torch.load('model4000000.pt'), None]

result, moves, board_history = play_against_computer(players, models)