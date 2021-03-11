import numpy as np

# Copied in large part from my math-programming repository
# Small edits made for compatibility with board format
# Fixed small logic issue where friendly victories were not always taken when possible
# Changed logic of preventing double attacks in the future
# This provides a good (but not ideal) "sparring partner" to train the neural net

# Check if choosing a certain square would immediately win
def checkVictory(board, y, x, player):
    state = board.copy()
    state[y, x] = player
    def win(row):
        return np.all(row == player)
    for row in state:
        if win(row):
            return True
    for col in state.transpose():
        if win(col):
            return True
    if win(state.diagonal()) or win(np.fliplr(state).diagonal()):
        return True
    return False

# Return the move to make for a particular board
# board: 3x3 numpy array
# player: 1 for Xs, -1 for Os
def getCPUInput(board, player): # Return (y, x, player) where player is 1=X or -1=O
    if not board[1, 1]:
        return 1, 1, player
    else:
        # Check for immediate friendly victories
        for r in range(3):
            for c in range(3):
                if not board[r, c]:
                    if checkVictory(board, r, c, player):
                        return r, c, player

        # Block immediate opponent victories
        for r in range(3):
            for c in range(3):
                if not board[r, c]:
                    if checkVictory(board, r, c, -player):
                        return r, c, player
                    
        # Check for future double-attacks
        doubleAttacks = np.zeros((3,3))
        for R in range(3):
            for C in range(3):
                if not board[R, C]:
                    newBoard = board.copy()
                    # Hypothetically place in spot
                    newBoard[R, C] = player
                    # "If I move in (R, C), then what can they do?"
                    for r in range(3):
                        for c in range(3):
                            if not newBoard[r, c]:
                                newBoard2 = newBoard.copy()
                                # Hypothetically place response
                                newBoard2[r, c] = -player
                                # "They would move to (r, c). Does this make a double attack?"
                                possibleAttacks = 0
                                for r2 in range(3):
                                    for c2 in range(3):
                                        if not newBoard[r2, c2]:
                                            # Track possible opponent victories after response
                                            if checkVictory(newBoard2, r2, c2, -player):
                                                possibleAttacks += 1
                                if possibleAttacks > 1:
                                    doubleAttacks[R, C] += 1

        # Prevent future double-attacks
        for r in range(3): # For each spot on the board...
            for c in range(3):
                # If it is an empty spot and 
                if not board[r, c] and doubleAttacks[r, c]:
                    return r, c, player

        # Take what you can get! Any empty spot works.
        for r in range(3):
            for c in range(3):
                if not board[r][c]:
                    return r, c, player

        # TODO: Implement seeking for double attacks

