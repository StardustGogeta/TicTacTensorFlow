# Taken from https://web.stanford.edu/~surag/posts/alphazero.html
# Taken from https://github.com/BryceGerst/PythonChessBot

import numpy as np
from math import sqrt

class MCTS:
    # We have Q[s][a], which returns the expected reward for making a move whose string representation
    # is a from the game board whose hash value is s.
    # We have N[s][a], which returns the number of times that we have explored making a move whose string
    # representation is a from the game board whose hash value is s.
    # Finally, we have P[s][a], which returns the initial probablility of making a move whose string
    # representation is a from the game board whose hash value is s. (aka the policy)
    # These will all be stored as dictionaries (hash tables) with the board hash as a key and an array
    # of size 4672 as values.
    def __init__(self):
        self.Q = {}
        self.N = {}
        self.P = {}
        self.visited = set()
        # visited and P should have the same number of elements in them

        # c_puct "is a hyperparameter that controls the degree of exploration"
        # hyperparameter means that it is external to the model itself
        self.c_puct = 0.2

    def search(self, s, game, nnet):
        if game.gameEnded(s): return -game.gameReward(s)

        s_key = str(s) # get hash from state to use as dictionary key

        if s_key not in self.visited:
            self.visited.add(s_key)

            #print(s.shape)
            s2 = np.array([s])
            #print(s2.shape)
            #pred = nnet.predict(s2)
            #print(pred)
            policy, v = nnet.predict(s2)
            self.P[s_key] = policy.reshape((3,3,2))
            self.Q[s_key] = np.zeros((3,3,2))
            self.N[s_key] = np.zeros((3,3,2))
            return -v
      
        max_u, best_a = -float("inf"), -1
        for a in game.getValidActions(s):
            a_key = (a[0], a[1], 0 if a[2] == 1 else 1) # Convert action to np index
            u = (self.Q[s_key][a_key] +
                self.c_puct*self.P[s_key][a_key]*
                sqrt(self.N[s_key].sum())/
                (1+self.N[s_key][a_key]))
            if u > max_u:
                max_u = u
                best_a = a
        a = best_a
        a_key = (a[0], a[1], 0 if a[2] == 1 else 1) # Convert action to np index
        
        sp = game.nextState(s, a)
        v = self.search(sp, game, nnet)

        self.Q[s_key][a_key] = (self.N[s_key][a_key]*self.Q[s_key][a_key] + v)/(self.N[s_key][a_key]+1)
        self.N[s_key][a_key] += 1
        return -v

    def pi(self, s):
        s_key = str(s) # get hash from state to use as dictionary key
        if s_key in self.P:
            return self.P[s_key]
        else:
            print('error, gamestate not found')
            return None

    def n(self, s):
        s_key = str(s) # get hash from state to use as dictionary key
        if s_key in self.N:
            return self.N[s_key]
        else:
            print('error, gamestate not found')
            return None

