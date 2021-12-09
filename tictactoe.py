"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    count = 0
    for row in board:
        for r in row:
            if r != EMPTY:
                count += 1
    if count % 2 == 0:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for r in range(3):
        for c in range(3):
            if board[r][c] == EMPTY:
                actions.add((r,c))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise Exception
    board_copy = copy.deepcopy(board)
    for r in range(3):
        for c in range(3):
            if (r,c) == action:
                board_copy[r][c] = player(board)
    return board_copy

    

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    #Horizontal
    for row in board:
        if row.count(X) == 3:
            return X
        if row.count(O) == 3:
            return O
    #Vertical
    for col in range(3):
        if board[0][col] == X and board[1][col] == X and board[2][col] == X:
            return X
        if board[0][col] == O and board[1][col] == O and board[2][col] == O:
            return O
    #DiagonalDown
    if board[0][0] == X and board[1][1] == X and board[2][2] == X:
        return X
    if board[0][0] == O and board[1][1] == O and board[2][2] == O:
        return O
    #DiagonalUp
    if board[2][0] == X and board[1][1] == X and board[0][2] == X:
        return X
    if board[2][0] == O and board[1][1] == O and board[0][2] == O:
        return O
    #NoWinner
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) == X or winner(board) == O:
        return True
    else:
        for r in range(3):
            for c in range(3):
                if board[r][c] == EMPTY:
                    return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0
 


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    def min_value(board):
        v = math.inf
        move = None
        if terminal(board):
            return None, utility(board)
        else:
            for action in actions(board):
                move_action, v_action =  max_value(result(board,action))
                if v_action < v:
                    move = action
                    v = v_action
                    if v == -1:
                        return move, v
        return move, v

    def max_value(board):
        v = -math.inf
        move = None
        if terminal(board):
            return None, utility(board)
        else:
            for action in actions(board):
                move_action, v_action = min_value(result(board, action))
                if v_action > v:
                    move = action
                    v = v_action
                    if v == 1:
                        return move, v
        return move, v

    if terminal(board):
        return None
    else:
        if player(board) == X: #Maximising player
            move, v = max_value(board)
            return move

        else:
            move, v = min_value(board)
            return move
