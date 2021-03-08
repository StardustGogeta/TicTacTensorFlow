import tensorflow as tf
from mcts import MCTS
from game import Game

numMCTSSims = 80 # number of times it iterates Monte Carlo tree search
threshold = 0.51 # win percentage threshold for neural net replacement
gameCount = 100 # games to play between competing neural nets
numIters = 10 # number of iterations
numEps = 5 # number of episodes

def policyIterSP(game):
    nnet = initNNet() # initialise random neural network
    examples = []    
    for i in range(numIters):
        for e in range(numEps):
            examples += executeEpisode(game, nnet) # collect examples from this game
        new_nnet = trainNNet(nnet, examples)                  
        frac_win = pit(new_nnet, nnet) # compare new net with previous net
        if frac_win > threshold: 
            nnet = new_nnet # replace with new net            
    return nnet

def executeEpisode(game, nnet):
    examples = []
    s = game.startState()
    mcts = MCTS()
    
    while True:
        a, examples = getOptimalAction(game, mcts, nnet, examples)
        s = game.nextState(s,a)
        if game.gameEnded(s):
            examples = assignRewards(examples, game.gameReward(s)) 
            return examples

def assignRewards(examples, reward):
    for example in examples[::-1]:
        example[2] = reward
        reward = -reward
    return examples

def getOptimalAction(game, mcts, nnet, examples=None):
    for _ in range(numMCTSSims):
        mcts.search(s, game, nnet)
    policy = mcts.pi(s)
    if examples is not None:
        examples.append([s, policy, None])
    a = None
    max_policy_val = -1
    for action in game.getValidActions(s):
        a_key = (action[0], action[1], 0 if action[2] == 1 else 1) # Convert action to np index
        policy_val = policy[a_key]
        if (policy_val > max_policy_val): # find the maximum likelihood move that is legal
            max_policy_val = policy_val
            a = action
    if examples is not None:
        return a, examples
    return a

def pit(new, old, game, nnet):
    newWins = 0
    for _ in range(gameCount // 2):
        mcts_new = MCTS()
        mcts_old = MCTS()
        
        s = game.startState()
        while True:
            ## New player goes first
            a = getOptimalAction(game, mcts_new, nnet)
            s = game.nextState(s,a)
            if game.gameEnded(s):
                newWins += (game.gameReward(s) + 1) / 2
                break # end the game simulation

            # Old player goes second
            a = getOptimalAction(game, mcts_old, nnet)
            s = game.nextState(s,a)
            if game.gameEnded(s):
                break # end the game simulation
        while True:
            ## Old player goes first
            a = getOptimalAction(game, mcts_old, nnet)
            s = game.nextState(s,a)
            if game.gameEnded(s):
                break # end the game simulation

            # New player goes second
            a = getOptimalAction(game, mcts_new, nnet)
            s = game.nextState(s,a)
            if game.gameEnded(s):
                newWins += (game.gameReward(s) + 1) / 2
                break # end the game simulation
     return newWins / gameCount

def initNNet():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(3,3)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5)
    ])
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

def trainNNet(nnet, examples):
    # examples are (state, policy, value)
    x_train, _, y_train = zip(*examples)
    model.fit(












    
