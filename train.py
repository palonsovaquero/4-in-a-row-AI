import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

from networks import DQN, update_weights
from utils import play_a_game

ngames = 30000

players = ["computer", "random"]

models = [DQN(), None] # Ex. torch.load('model_3.pt'), DQN(), None -> random
#models = [torch.load('model1.pt'), torch.load('model1.pt')] # This trains model1 by playing against itself

optimizer = torch.optim.SGD(models[0].parameters(), lr=0.1)
loss_function = nn.MSELoss()
discount_factor = 0.9

results = np.zeros(3)
results_total = np.zeros(3)
lst_good_moves = [[],[]]
perc_good_moves = []

rewards = [1, 1, -1] # draw, win, loss

for i in range(ngames):

    ### Play a game ###

    result, moves, board_history, good_moves = play_a_game(players, models)

    ### Update the model weights ###

    models[0] = update_weights(rewards[result], board_history[0], moves[0], models[0], optimizer, loss_function, discount_factor)

    ### Save the statistics ###

    lst_good_moves[0].append(good_moves[0]) # bad moves
    lst_good_moves[1].append(good_moves[1]) # good moves
    results[result]+=1
    results_total[result]+=1

    ### Print the results and save the model every 10000 games ###

    if (i+1)%10000==0:
        print("\nresults after", i+1, "games:\n", "draw --------->", 100*results[0]/10000, "%\n player1 wins ->", 100*results[1]/10000, "%\n player2 wins ->", 100*results[2]/10000, "%")
        print("\npercentage of good moves after", i+1, "games: ", np.mean(np.array(lst_good_moves[1])))
        results = np.zeros(3)

### Save the final model ###

torch.save(models[0], 'final_model.pt')
