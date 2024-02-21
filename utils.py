import numpy as np
import torch

#-----------------------------------------------------------------------------------------------------------------

def create_board(y, x):
    board = np.zeros((y, x))
    return board

#-----------------------------------------------------------------------------------------------------------------

def play_a_game(players, models): # players is a list of [computer, computer], or [computer, human], at least one must be "computer"
    board = create_board(6, 7)
    move_number = 0
    moves = [[],[]]
    board_history = [[],[]]
    good_moves = np.zeros(2)
    while check_gameover(board) == -1:
        board_history[move_number % 2].append(board.flatten())
        move, board, move_good = make_a_move(board, move_number % 2, players, models)
        if move_good!=-1:
            good_moves[move_good]+=1 # good_moves[0]->bad moves,  good_moves[1]->good moves
        moves[move_number % 2].append(move)
        move_number += 1
    return check_gameover(board), moves, board_history, good_moves/(good_moves[0]+good_moves[1])

#-----------------------------------------------------------------------------------------------------------------

def play_against_computer(players, models): # players is a list of [computer, computer], or [computer, human], at least one must be "computer"
    board = create_board(6, 7)
    move_number = 0
    moves = [[],[]]
    board_history = [[],[]]
    while check_gameover(board) == -1:
        board_history[move_number % 2].append(board.flatten())
        move, board, move_good = make_a_move(board, move_number % 2, players, models)
        moves[move_number % 2].append(move)
        move_number += 1
        print("Move: ", move_number)
        print(board)
    return check_gameover(board), moves, board_history

#-----------------------------------------------------------------------------------------------------------------

def check_gameover(board):  # Return: -1 -> game is not over, 0 -> draw, 1 -> player1 wins, 2 -> player2 wins.
    if len(np.where(board == 0)[0]) == 0:
        return 0
    else:
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] in [1, 2]:
                    try:
                        if board[i-1, j] == board[i, j] and board[i-2, j] == board[i, j] and board[i-3, j] == board[i, j] and i-3 in range(0, board.shape[0]):
                            #print("Player ", int(board[i, j]), " wins!")
                            return int(board[i, j])
                    except:
                        a = 0
                    try:
                        if board[i+1, j] == board[i, j] and board[i+2, j] == board[i, j] and board[i+3, j] == board[i, j] and i+3 in range(0, board.shape[0]):
                            #print("Player ", int(board[i, j]), " wins!")
                            return int(board[i, j])
                    except:
                        a = 0
                    try:
                        if board[i, j-1] == board[i, j] and board[i, j-2] == board[i, j] and board[i, j-3] == board[i, j] and j-3 in range(0, board.shape[0]):
                            #print("Player ", int(board[i, j]), " wins!")
                            return int(board[i, j])
                    except:
                        a = 0
                    try:
                        if board[i, j+1] == board[i, j] and board[i, j+2] == board[i, j] and board[i, j+3] == board[i, j] and j+3 in range(0, board.shape[0]):
                            #print("Player ", int(board[i, j]), " wins!")
                            return int(board[i, j])
                    except:
                        a = 0
                    try:
                        if board[i-1, j-1] == board[i, j] and board[i-2, j-2] == board[i, j] and board[i-3, j-3] == board[i, j] and i-3 in range(0, board.shape[0]) and j-3 in range(0, board.shape[0]):
                            #print("Player ", int(board[i, j]), " wins!")
                            return int(board[i, j])
                    except:
                        a = 0
                    try:
                        if board[i+1, j+1] == board[i, j] and board[i+2, j+2] == board[i, j] and board[i+3, j+3] == board[i, j] and i+3 in range(0, board.shape[0]) and j+3 in range(0, board.shape[0]):
                            #print("Player ", int(board[i, j]), " wins!")
                            return int(board[i, j])
                    except:
                        a = 0
                    try:
                        if board[i-1, j+1] == board[i, j] and board[i-2, j+2] == board[i, j] and board[i-3, j+3] == board[i, j] and i-3 in range(0, board.shape[0]) and j+3 in range(0, board.shape[0]):
                            #print("Player ", int(board[i, j]), " wins!")
                            return int(board[i, j])
                    except:
                        a = 0
                    try:
                        if board[i+1, j-1] == board[i, j] and board[i+2, j-2] == board[i, j] and board[i+3, j-3] == board[i, j] and i+3 in range(0, board.shape[0]) and j-3 in range(0, board.shape[0]):
                            #print("Player ", int(board[i, j]), " wins!")
                            return int(board[i, j])
                    except:
                        a = 0
    return -1

#-----------------------------------------------------------------------------------------------------------------

def make_a_move(board, player_index, players, models):
    ilegal_moves = get_ilegal_moves(board) # 0->good, 1->ilegal
    if players[player_index] == "random":
        good_move = 0
        while good_move == 0:
            column = np.random.randint(0, board.shape[1])
            if illegal_move(board, column) == False:
                good_move = 1
        for i in range(board.shape[0]-1, -1, -1):
            if board[i, column] == 0:
                board[i, column] = player_index+1
                row = i
                break
    if players[player_index] == "computer":

        good_move = 0
        good_move_temp = 0

        q_values = models[player_index](torch.from_numpy(board.flatten()))

        move = torch.argmax(q_values).item()
        row = move // 7
        column = move % 7

        if ilegal_moves[move]==0:
            good_move = 1
            good_move_temp = 1
            board[row, column] = player_index+1
        if good_move == 0:
            while good_move == 0: #if the network chose an illegal move. IMPROVE!
                column = np.random.randint(0, board.shape[1])
                if illegal_move(board, column) == False:
                    good_move = 1
            for i in range(board.shape[0]-1, -1, -1):
                if board[i, column] == 0:
                    board[i, column] = player_index+1
                    row = i
                    break
        return row*board.shape[1]+column, board, good_move_temp
    if players[player_index] == "human":
        column = int(input("file (1 to 7): "))-1
        if illegal_move(board, column) == True:
            print("Illegal move!")
            #make_a_move(board, player_index, players, model)
        else:
            for i in range(board.shape[0]-1, -1, -1):
                if board[i, column] == 0:
                    board[i, column] = player_index+1
                    row = 1
                    break
    return row*board.shape[1]+column, board, -1
    #return column, board

#-----------------------------------------------------------------------------------------------------------------

def illegal_move(board, move):
    for i in range(board.shape[0]-1, -1, -1):
        if board[i, move] == 0:
            return False
    return True

#-----------------------------------------------------------------------------------------------------------------

def get_ilegal_moves(board):
    ilegal_moves = np.zeros((board.shape[0], board.shape[1]))
    for i in range(board.shape[1]):
        for j in range(board.shape[0]):
            if j==board.shape[0]-1:
                if board[j][i]!=0:
                    ilegal_moves[j][i]=1
            else:
                if board[j+1][i]==0 or board[j][i]!=0:
                    ilegal_moves[j][i]=1
    return ilegal_moves.flatten()

#-----------------------------------------------------------------------------------------------------------------

