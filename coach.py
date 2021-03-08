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
        a, examples = getOptimalAction(game, s, mcts, nnet, examples)
        s = game.nextState(s,a)
        if game.gameEnded(s):
            examples = assignRewards(examples, game.gameReward(s)) 
            return examples

def assignRewards(examples, reward):
    for example in examples[::-1]:
        example[2] = reward
        reward = -reward
    return examples

def getOptimalAction(game, s, mcts, nnet, examples=None):
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
    # creates and returns new neural network

    # standard sequential model:
##    model = tf.keras.models.Sequential([
##        tf.keras.layers.Flatten(input_shape=(3,3)),
##        tf.keras.layers.Dense(64, activation='relu'),
##        tf.keras.layers.Dropout(0.2),
##        tf.keras.layers.Dense(5)
##    ])

    # functional model:
##    inputs = tf.keras.Input(shape=(3,3))
##    x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
##    outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
##    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # modified functional:
    # https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
    Input = tf.keras.Input
    Dense = tf.keras.layers.Dense
    Flatten = tf.keras.layers.Flatten
    Concatenate = tf.keras.layers.Concatenate
    Lambda = tf.keras.layers.Lambda
    Model = tf.keras.Model
    
##    boardInput = Input(shape=(3,3))
##    policyInput = Input(shape=(3,3,2))
##
##    x = Dense(64, activation='relu')(boardInput)
##    x = Dense(2, activation='relu')(x)
##    x = Model(inputs=boardInput, outputs=x)
##
##    y = Dense(64, activation='relu')(policyInput)
##    y = Dense(2, activation='relu')(y)
##    y = Model(inputs=policyInput, outputs=y)
##
##    combined = Concatenate([x.output, y.output])


    # Goes like this:
    # x (board) --| --> y (policy)
    #             | --> z (value)
    #
    boardInput = Input(shape=(3,3))
    #print(boardInput.shape)
    x = Dense(64)(boardInput)
    x = Dense(20, activation='relu')(x)
    x = Flatten()(x)

    y = Dense(18, name="dense_policy_1")(x)
    #y.reshape((3,3,2))
    #y = Model(inputs=boardInput, outputs=y)

    z = Dense(10, name="dense_value_1")(x)
    z = Dense(1, name="dense_value_2")(z)
    #z = Model(inputs=boardInput, outputs=z)

    #print(y.summary())
    #print(z.summary())

    a = Concatenate()([y, z])

    #output = Lambda(lambda t: Concatenate()(t))([y, z])
    
    model = Model(inputs=boardInput, outputs=a)
    print(model.summary())

    # when we train it, we want to feed in examples of good moves and bad moves
    # the model should be able to take in a board state and return an optimal policy, and if it is winning or losing
    
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

def trainNNet(nnet, examples):
    # examples are of the form (state, policy, value)
    x_train, p_train, y_train = zip(*examples)
    model.fit([x_train, p_train], y_train, epochs=5, verbose=1)

game = Game()
policyIterSP(game)



