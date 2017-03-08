# The experience replay, policy and value function is all encapsulated up into this agent class. The agent
# select actions (enact the eGreedy policy), observe the consequences (store the S,A,R,S' info in experience replay)
# and then reflect on all of this (train its Q-function). 

from policy import eGreedy
from value_function import deepQNetwork
from experience_replay import experienceReplay, memoryNode
import numpy as np
def argmax(b):
    maxVal = None
    maxData = None
    for i,a in enumerate(b):
        if a>maxVal:
            maxVal = a
            maxData = i
    return maxData
class agent:
    
    def __init__(self, stateDim, actions, learningRate=0.05, gamma=0.99, epsilon=1, memorySize=150000):
        self.gamma = gamma
        self.stateDim = stateDim
        self.actions = actions
        self.policy = eGreedy(epsilon)
        self.Q = deepQNetwork(learningRate, stateDim, len(actions))
        self.Q_est = deepQNetwork(learningRate, stateDim, len(actions))
        self.experience = experienceReplay(memorySize)
        
    def act(self,state):
        return self.policy.enact(self.actions, self.Q.predict(state[np.newaxis,:]))
        
    def observe(self, state, action, reward, nextState):
        self.experience.remember(state, action, reward, nextState)

    # By which I mean run through some experience and update the Q function accordingly
    def reflect(self, iteration, batchSize = 250):
        targets = np.zeros((batchSize,len(self.actions)))
        states = np.zeros((batchSize,self.stateDim))
                        
        for (i, memory) in enumerate(self.experience.recall(self.Q, self.Q_est, batchSize)):
    
            targets[i] = self.Q_est.predict(memory.S[np.newaxis])
    
            # if the agent moves to the terminal state then the return is exactly the reward
            if memory.next.S is None:
                targets[i,memory.A] = memory.R
            # otherwise we bootstrap the return by observing the current reward and adding it to the value of the next state-greedy action 
            else:
                targets[i,memory.A] = memory.R + self.gamma * self.Q_est.predict(memory.next.S[np.newaxis])[0][argmax(self.Q.predict(memory.next.S[np.newaxis])[0])]
            states[i] = memory.S
                  
        # in case the experience replay wasn't able to serve up enough memories, we need to trim the matrices                  
        states.resize((i+1,self.stateDim))
        targets.resize((i+1,len(self.actions)))

        # and finally we pass this to the Q function for fitting
        self.Q.fit(states, targets)
        if iteration % 4 == 1:
            weights = self.Q.model.get_weights()#: returns a list of all weight tensors in the model, as Numpy arrays.
            self.Q_est.model.set_weights(weights)