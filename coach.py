import tensorflow as tf
from mcts import MCTS
from game import Game
import numpy as np
import time, random

numMCTSSims = 6 # number of times it iterates Monte Carlo tree search
threshold = 0.51 # win percentage threshold for neural net replacement
gameCount = 10 # games to play between competing neural nets
numIters = 10 # number of iterations
numEps = 4 # number of episodes

loss_fns = [tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.MeanAbsoluteError()]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

AUTOSAVE = True
OLD_NNET_ALLOWED = True

def policyIterSP(game):
    if OLD_NNET_ALLOWED and input("Do you want to load the previous neural net? (y/n) ") == "y":
        nnet = tf.keras.models.load_model("./models/my_nnet")
    else:
        nnet = initNNet() # initialise random neural network
        saveNNet(nnet)
    print("Starting training")
    examples = []    
    for i in range(numIters):
        for e in range(numEps):
            examples += executeEpisode(game, nnet) # collect examples from this game
        new_nnet = trainNNet(nnet, examples)
        print(f"Done training iteration {i}")
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
    start = time.perf_counter()
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
        if (policy_val > max_policy_val or a is None): # find the maximum value move that is legal
            max_policy_val = policy_val
            a = action
    #print(f"Found optimal action\t\t[{time.perf_counter() - start} sec]")
    if examples is not None:
        return a, examples
    if a is None:
        print(game, s, mcts, nnet, game.getValidActions(s))
        raise ZeroDivisionError
    return a

def pit(new, old, game):
    start = time.perf_counter()
    newWins = 0
    mcts_new = MCTS()
    mcts_old = MCTS()
    for _ in range(gameCount // 2):
##        mcts_new = MCTS()
##        mcts_old = MCTS()
        
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

##        mcts_new = MCTS()
##        mcts_old = MCTS()
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
    print(f"Success rate: {newWins / gameCount:.2f}\t\t[{time.perf_counter() - start} sec]")
    return newWins / gameCount

def initNNet():
    # creates and returns new neural network
    # modified functional model:
    # https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
    Input = tf.keras.Input
    Dense = tf.keras.layers.Dense
    Reshape = tf.keras.layers.Reshape
    Flatten = tf.keras.layers.Flatten
    Dropout = tf.keras.layers.Dropout
    Concatenate = tf.keras.layers.Concatenate
    Lambda = tf.keras.layers.Lambda
    Model = tf.keras.Model
    
    # Goes like this:
    # x (board) --| --> y (policy)
    #             | --> z (value)
    #
    boardInput = Input(shape=(3,3))
    x = Dense(8, activation='relu')(boardInput)
    x = Dense(20, activation='relu')(x)
    x = Flatten(name="flattened")(x)
    x = Dropout(0.3, name="dropped")(x)

    y = Dense(18, activation='relu', name="dense_policy_1")(x)
    y = Reshape((3,3,2), name="reshape_policy")(y)

    z = Dense(10, activation='relu', name="dense_value_1")(x)
    z = Dense(1, name="dense_value_2")(z)
    
    model = Model(inputs=boardInput, outputs=[y,z])
    print(model.summary())

    # when we train it, we want to feed in examples of good moves and bad moves
    # the model should be able to take in a board state and return an optimal policy, and if it is winning or losing
    
    model.compile(optimizer=optimizer, loss=loss_fns, metrics=['accuracy'])
    return model

def trainNNet(nnet, examples):
    new_nnet = tf.keras.models.clone_model(nnet)
    new_nnet.compile(optimizer=optimizer, loss=loss_fns, metrics=['accuracy'])
    
    # examples are of the form (state, policy, value)
    x_train, p_train, y_train = zip(*examples)
    new_nnet.fit(np.array(x_train), [np.array(p_train), np.array(y_train)], epochs=10, verbose=0)
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

def playAgainstRandom(nnet, game, numGames):
    #numGames = 100
    netWins = 0
    netTies = 0
    
    for i in range(numGames // 2):
        mcts = MCTS()
        s = game.startState()
        while True:
            # Random player goes first
            a = random.choice(game.getValidActions(s))
            s = game.nextState(s,a)
            if game.gameEnded(s):
                reward = (-game.gameReward(s) + 1) / 2
                if reward == 1:
                    netWins += 1
                elif reward == 1/2:
                    netTies += 1
                break # end the game simulation

            # Neural net goes second
            a = getOptimalAction(game, s, mcts, nnet)
            s = game.nextState(s,a)
            if game.gameEnded(s):
                reward = (game.gameReward(s) + 1) / 2
                if reward == 1:
                    netWins += 1
                elif reward == 1/2:
                    netTies += 1
                break # end the game simulation
        #print("\n-----\nFINAL BOARD\n-----\n", s, "\n")
        
        mcts = MCTS()
        s = game.startState()
        while True:
            # Neural net goes first
            a = getOptimalAction(game, s, mcts, nnet)
            s = game.nextState(s,a)
            if game.gameEnded(s):
                reward = (game.gameReward(s) + 1) / 2
                if reward == 1:
                    netWins += 1
                elif reward == 1/2:
                    netTies += 1
                break # end the game simulation

            # Random player goes second
            a = random.choice(game.getValidActions(s))
            s = game.nextState(s,a)
            if game.gameEnded(s):
                reward = (-game.gameReward(s) + 1) / 2
                if reward == 1:
                    netWins += 1
                elif reward == 1/2:
                    netTies += 1
                break # end the game simulation
        #print("\n-----\nFINAL BOARD\n-----\n", s, "\n")
        print(f"{2*(i+1)} of {numGames} games complete...")
    print(f"Neural net won {netWins}, tied {netTies}, and lost {numGames-netWins-netTies} of {numGames} games against random.")

game = Game()
policyIterSP(game)

best_nnet = tf.keras.models.load_model("./models/my_nnet")
#print(best_nnet.summary())
playAgainstRandom(best_nnet, game, 200)
#playAgainstHuman(best_nnet, game)

