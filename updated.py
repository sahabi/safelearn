# This algorith is based around DQN, parameterised like so:

# Q-function is a dense network with 2x100 node hidden layers
# experience replay contained the most recent 10000 state, action, reward triplets.
# learning took place after every episode using a minibatch size of 100
# learning rate = 0.01
# gamma = 0.99 
# eGreediness = 0.05

from collections import deque
import env_m
#from gym import wrappers
import numpy as np
from agent import agent

LEFT = 0
RIGHT = 1
MAX_TIMESTEPS = 500

blob = agent(4,[i for i in range(0,8)], epsilon=0.05)
env = env_m.Env()
#env = wrappers.Monitor(env, '/tmp/cartpole-experiment-v1',force=True)

t = 0
avgreward = deque([],100)
trials = 10000
for i_episode in range(trials):
    
    S = env.reset()

    done = False   
    t = 0
    tot_R = 0
    
    while not done:
        t += 1
        A = blob.act(S)
        S_dash, R, done = env.step(A)
        
        blob.observe(S,A,R,S_dash)
        tot_R += R
        
        S = np.copy(S_dash)
        
    # every now and then stop, and think things through:
    blob.reflect()
        
    # when the episode ends the agent will have hit a terminal state so give it a zero reward
    if t < MAX_TIMESTEPS:
        blob.observe(S,A,0.,None)
    else:
        blob.observe(S,A,1.,None)
            
    avgreward.append(tot_R)
    
    print("episode: {}, average reward: {}, Reward: {}".format(i_episode,np.mean(avgreward),tot_R))

env.close() 