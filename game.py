import numpy as np
import copy

class Game:
    @staticmethod
    def startState():
        # returns state (3x3 array)
        # 0 is blank
        # 1 is X
        # -1 is O
        return np.zeros((3,3))

    @staticmethod
    def nextState(state, action):
        # return state
        # action is tuple (y, x, player)
        # player is 1 or -1
        newState = state.copy()
        (y, x, player) = action
        newState[y, x] = player
        return newState

    @staticmethod
    def getValidActions(state):
        # return list of valid actions
        xCount = 0
        oCount = 0
        blanks = []
        for y, row in enumerate(state):
            for x, entry in enumerate(row):
                if entry == 1:
                    xCount += 1
                elif entry == -1:
                    oCount += 1
                else:
                    blanks.append((y, x))
        player = 1 if xCount == oCount else -1
        return [(y, x, player) for (y, x) in blanks]

    @staticmethod
    def gameEnded(state):
        # return boolean of whether game is ended
        # check for filled board
        if np.count_nonzero(state) == 9:
            return True
        
        def triple(row):
            # returns if a row is 3 Xs or 3 Os
            return np.all(row == 1) or np.all(row == -1)
        for row in state:
            if triple(row):
                return True
        for col in state.transpose():
            if triple(col):
                return True
        if triple(state.diagonal()) or triple(np.fliplr(state).diagonal()):
            return True
        return False

    @staticmethod
    def gameReward(state):
        # returns -1 if win for current player in this state, 1 if loss, 0 if tie
        # note that win at the start of your turn is impossible
        xCount = 0
        oCount = 0
        for y, row in enumerate(state):
            for x, entry in enumerate(row):
                if entry == 1:
                    xCount += 1
                elif entry == -1:
                    oCount += 1
        player = 1 if xCount == oCount else -1
        
        def lose(row):
            # returns if a row is 3 in a row
            return np.all(row == -player)
        
        for row in state:
            if lose(row):
                return 1
        for col in state.transpose():
            if lose(col):
                return 1
        if lose(state.diagonal()) or lose(np.fliplr(state).diagonal()):
            return 1
        return 0 # If the player doesn't win, it must be a draw!
