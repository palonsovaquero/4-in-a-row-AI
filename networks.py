import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import get_ilegal_moves

#-----------------------------------------------------------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(42, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 42)
    def forward(self, x):
        x = x.float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

#-----------------------------------------------------------------------------------------------------------------

def update_weights(reward, board_history, moves, model, optimizer, loss_function, discount_factor):
    # Last move, with reward.
    #model = backprop(board_history[], moves[], model, optimizer, loss, reward)
    #print("moves = ", moves)
    for i in range(len(moves)):
        if reward>0:
            model = backprop(board_history[len(moves)-1-i], moves[len(moves)-1-i], model, optimizer, loss_function, max(reward*discount_factor**(i+1), 0.3))
        else:
            model = backprop(board_history[len(moves)-1-i], moves[len(moves)-1-i], model, optimizer, loss_function, min(reward*discount_factor**(i+1), -0.3))
    return model

#-----------------------------------------------------------------------------------------------------------------

def backprop(board, move, model, optimizer, loss_function, reward):
    output = model(torch.from_numpy(board))#.detach()
    target = output.clone()
    target[:] = 0.2
    ilegal_moves = get_ilegal_moves(np.resize(board,(6,7)))
    target[move] = reward
    target[np.where(ilegal_moves==1)]=0.0
    loss = loss_function(output, target)
    optimizer.zero_grad()
    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()
    return model

#-----------------------------------------------------------------------------------------------------------------