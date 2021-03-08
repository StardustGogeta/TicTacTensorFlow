import tensorflow as tf
from mcts import MCTS
from game import Game
import numpy as np

numMCTSSims = 20 # number of times it iterates Monte Carlo tree search
threshold = 0.51 # win percentage threshold for neural net replacement
gameCount = 20 # games to play between competing neural nets
numIters = 5 # number of iterations
numEps = 5 # number of episodes

AUTOSAVE = True

def policyIterSP(game):
    if input("Do you want to load the previous neural net? (y/n) ") == "y":
        nnet = tf.keras.models.load_model("./models/my_nnet")
    else:
        nnet = initNNet() # initialise random neural network
        saveNNet(nnet)
    examples = []    
    for i in range(numIters):
        for e in range(numEps):
            examples += executeEpisode(game, nnet) # collect examples from this game
        new_nnet = trainNNet(nnet, examples)
        print("Done training")
        frac_win = pit(new_nnet, nnet, game) # compare new net with previous net
        print("Done pitting")
        if frac_win > threshold:
            nnet = new_nnet # replace with new net
            saveNNet(nnet)
            
    return nnet

def saveNNet(nnet):
    if AUTOSAVE or input("Do you want to save the new neural net? (y/n) ") == "y":
        nnet.save("./models/my_nnet")
        print("Done saving")

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
    max_policy_val = -2
    for action in game.getValidActions(s):
        a_key = (action[0], action[1], 0 if action[2] == 1 else 1) # Convert action to np index
        policy_val = policy[a_key]
        if (policy_val > max_policy_val): # find the maximum value move that is legal
            max_policy_val = policy_val
            a = action
    if examples is not None:
        return a, examples
    if a is None:
        print(game, s, mcts, nnet, game.getValidActions(s))
        raise ZeroDivisionError
    return a

def pit(new, old, game):
    newWins = 0
    for _ in range(gameCount // 2):
        mcts_new = MCTS()
        mcts_old = MCTS()
        
        s = game.startState()
        while True:
            ## New player goes first
            a = getOptimalAction(game, s, mcts_new, new)
            s = game.nextState(s,a)
            if game.gameEnded(s):
                # gameReward is +1 if new player has won
                # gameReward is -1 if new player has lost
                # gameReward is 0 if new player has tied
                newWins += (game.gameReward(s) + 1) / 2
                break # end the game simulation

            # Old player goes second
            a = getOptimalAction(game, s, mcts_old, old)
            s = game.nextState(s,a)
            if game.gameEnded(s):
                newWins += (-game.gameReward(s) + 1) / 2
                break # end the game simulation

        mcts_new = MCTS()
        mcts_old = MCTS()
        s = game.startState()
        while True:
            ## Old player goes first
            a = getOptimalAction(game, s, mcts_old, old)
            s = game.nextState(s,a)
            if game.gameEnded(s):
                # gameReward is -1 if new player has won
                # gameReward is +1 if new player has lost
                # gameReward is 0 if new player has tied
                newWins += (-game.gameReward(s) + 1) / 2
                break # end the game simulation

            # New player goes second
            a = getOptimalAction(game, s, mcts_new, new)
            s = game.nextState(s,a)
            if game.gameEnded(s):
                newWins += (game.gameReward(s) + 1) / 2
                break # end the game simulation
    print(newWins / gameCount)
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
    x = Dense(8)(boardInput)
    #x = Dense(20, activation='relu')(x)
    x = Flatten(name="flattened")(x)

    y = Dense(18, name="dense_policy_1")(x)
    #y.reshape((3,3,2))
    #y = Model(inputs=boardInput, outputs=y)

    z = Dense(10, name="dense_value_1")(x)
    z = Dense(1, name="dense_value_2")(z)
    #z = Model(inputs=boardInput, outputs=z)

    #print(y.summary())
    #print(z.summary())

    #a = Concatenate()([y, z])

    #output = Lambda(lambda t: Concatenate()(t))([y, z])
    
    model = Model(inputs=boardInput, outputs=[y,z])
    print(model.summary())

    # when we train it, we want to feed in examples of good moves and bad moves
    # the model should be able to take in a board state and return an optimal policy, and if it is winning or losing
    
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model

def trainNNet(nnet, examples):
    new_nnet = tf.keras.models.clone_model(nnet)
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    new_nnet.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # examples are of the form (state, policy, value)
    #print("ex1:", examples[0])
##    np_ex = np.array(examples)
##    for ex in np_ex:
##        if ex[2] is None:
##            print("FAILED!")
##    print("ex shape:", np.array(examples).shape)
##    print("ex shape:", np.array(examples[0]).shape)
    x_train, p_train, y_train = zip(*examples)
##    for ex in examples:
##        print("---\nCASE\n",ex[0], ex[2], "\n\n")
##    raise ValueError
    in_train = tf.convert_to_tensor(x_train)
##    print(in_train.shape)
##    p = np.array(p_train)
##    print(p.shape)
    #y = np.array(y_train)
    #print(y.shape)
    out_train = tf.convert_to_tensor(y_train)
    new_nnet.fit(in_train, out_train, epochs=10, verbose=0)
    return new_nnet

def playAgainstHuman(nnet, game):
    mcts = MCTS()
    s = game.startState()
    while True:
        # Human player goes first
        print("\n-----\nBOARD\n-----\n", s, "\n")
        a = eval(input("Give human move (y, x, 1): "))
        s = game.nextState(s,a)
        if game.gameEnded(s):
            break # end the game simulation

        # Neural net goes second
        a = getOptimalAction(game, s, mcts, nnet)
        s = game.nextState(s,a)
        if game.gameEnded(s):
            break # end the game simulation
    print("\n-----\nFINAL BOARD\n-----\n", s, "\n")
    
    mcts = MCTS()
    s = game.startState()
    while True:
        # Neural net goes first
        a = getOptimalAction(game, s, mcts, nnet)
        s = game.nextState(s,a)
        if game.gameEnded(s):
            break # end the game simulation

        # Human player goes second
        print("\n-----\nBOARD\n-----\n", s, "\n")
        a = eval(input("Give human move (y, x, -1): "))
        s = game.nextState(s,a)
        if game.gameEnded(s):
            break # end the game simulation
    print("\n-----\nFINAL BOARD\n-----\n", s, "\n")

game = Game()
policyIterSP(game)

#best_nnet = tf.keras.models.load_model("./models/my_nnet")
#playAgainstHuman(best_nnet, game)

