# 4-in-a-row-AI

AI model that learns how to play 4 in a row.

4 in a row is a game that consists  on a board of size 6x7. Each turn, a player drops a coin on one of the 7 columns available, unless the column is already full. Following gravity, the coin is placed on the lowest unnocupied row.

The goal is to make 4 coins in a row, either vertically, horizontally, or diagonally.

In this project I use a recurrent learning algorithm known as Deep Q-Network (DQN) to train a model to play 4 in a row. A Deep Q-network is a combination of deep learning with Q-learning. Q-learning is a type of reinforcement learning that defines a Q-function that determines the expected future rewards for each action given a particular state. In a Deep Q-network, the neural network approximates the Q function, which in many cases is very complex and very costly to calculate.

To train the model, run:

<pre>
<code>
```bash
python train.py
```
</code>
</pre>


You can play with several parameters, such as the number of games for training, the number of layers in the model (defined in "networks.py"), the learning rate, or the rewards (I reward 1 for win or draw, and -1 for a loss).

By default, the DQN will play against random, and opponent making random moves. However, you can further train your model by playing against itself, by changing "players = ["computer", "random"]" to "players = ["computer", "computer"]" and "models = [DQN(), None]" to "models = [torch.load('model1.pt'), torch.load('model1.pt')]" in the "train.py" file.

In this repostory I attach a simple DQN model that has been trained in 2 million games against random. The results are quite favorable for the DQN. After training for 2 million games, the DQN model can beat random in more than 98% of the games. However, it is still relatively simple for a human to beat the DQN model.

If you want to test your skills against the model, run play_against_computer.py .
