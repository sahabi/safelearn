# OpenGym Seaquest-v0
# -------------------
#
# This code demonstrates a Double DQN network with Priority Experience Replay
# in an OpenGym Seaquest-v0 environment.
#
# Made as part of blog series Let's make a DQN, available at: 
# https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
# 
# author: Jaromir Janisch, 2016

#--- enable this to run on GPU
import os    
#os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"  
#---
from collections import deque
import numpy as np
from agent_pr import agent
import matplotlib
matplotlib.use('WxAgg')
from matplotlib import pyplot as plt
from dialogue import Widget
import random, numpy, math, scipy
from SumTree import SumTree
import env_road as env_m
import pickle

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network

    def _createModel(self):
        model = Sequential()
        model.add(Dense(1, input_dim=4))
        model.add(Activation('relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(40, activation='relu'))
        #self.model.add(Dense(16, activation='relu'))
        model.add(Dense(8,activation='linear')) 
        model.compile(lr=LR, optimizer='rmsprop', loss='mse')

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=32, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, 4), target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

    def predict_(self, states):        
        return self.model_.predict(states,batch_size=1)

    def predict_all(self, states):       
        return self.model_.predict(states, batch_size=32)


#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = .01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            # print 'sample {}: {}'.format(idx, data)
            batch.append( (idx, data) )
        
        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

#-------------------- AGENT ---------------------------


class Agent:
    
    steps = 0
    
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.exp = 0
        self.memory = Memory(MEMORY_CAPACITY)
        self.epsilon = MAX_EPSILON
        self.update_ep = 0
        self.brain = Brain(stateCnt, actionCnt)
        # self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample, iteration=0):  # in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)])
        #self.memory.add(errors[0], sample)
        self.memory.add(self.memory.tree.max_p, sample)
        self.exp += 1

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
           self.brain.updateTargetModel()
           self.update_ep = iteration

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
        return self.update_ep

    def _getTargets(self, batch):
        no_state = numpy.zeros(4)

        states = numpy.array([ o[1][0] for o in batch ])
        states_ = numpy.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])

        p = agent.brain.predict(states)

        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)

        x = numpy.zeros((len(batch),4))
        y = numpy.zeros((len(batch), self.actionCnt))
        errors = numpy.zeros(len(batch))
        
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)

class RandomAgent:
    
    

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt
        self.brain = Brain(8, actionCnt)
        self.exp = 0
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample, iteration=0):  # in (s, a, r, s_) format
        #error = abs(sample[2])  # reward
        #error = sample[2]
        error = self.memory.tree.max_p
        self.memory.add(error, sample)
        self.exp += 1

    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, viz=False):
        self.env = env_m.Env(viz=viz)

    def run(self, agent, viz_flag=False, iteration=0,ep_id=0):
        if viz_flag:
            env.viz = True
        else:
            env.viz = False          
        s = self.env.reset(agent.brain,iteration,viz_flag)

        R = 0
        while True:         
            a = agent.act(s)

            r = 0
            s_, r, done = self.env.step(a)

            if done: # terminal state
                s_ = None
            iteration += 1
            update_ep = agent.observe( (s, a, r, s_), ep_id )        

            s = s_
            R += r
            agent.replay() 
            

            if done:
                #agent.replay() 
                break

        #print("Total reward:", R)
        return (R,update_ep)

#-------------------- MAIN ----------------------------
MEMORY_CAPACITY = 10000
BATCH_SIZE = 7
GAMMA = 0.99
MAX_EPSILON = 1
MIN_EPSILON = 0.05
EXPLORATION_STOP = 2500   # at this step epsilon will be 0.01
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay
UPDATE_TARGET_FREQUENCY = 50
LR = .1

env = Environment()

stateCnt  = (8, 1)
actionCnt = 8

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

notify_value = -1
t = 0
avgreward = deque([],20)
trials = 100000
fig, ax = plt.subplots(1, 1)
ax.set_aspect('auto')
ax.set_xlim(0, 5000)
ax.set_ylim(-2, 2)
ax.set_ylabel('Rewards')
ax.set_xlabel('Episodes')
ax.hold(True)
x = deque([],500)
x.append(0)
y = deque([],500)
y.append(-1)
precision = 2

plt.show(False)
plt.draw()
maxsofar = -2
max_, = ax.plot((0, 5000), (maxsofar, maxsofar), 'k-')
thresh, = ax.plot((0, 5000), (notify_value, notify_value), 'k-')
upep, = ax.plot((0, 0), (-1, 1), 'k-', color = 'r') 
if True:
    # cache the background
    background = fig.canvas.copy_from_bbox(ax.bbox)

points = ax.plot(x, y, 'o')[0]
#rand = 50000
rand_agent = True
try:
    agent.memory = pickle.load( open( "memory_abs{}.p".format(MEMORY_CAPACITY), "rb" ) )
except:
    print("Initialization with random agent...")
    while randomAgent.exp < MEMORY_CAPACITY and rand_agent:
        env.run(randomAgent)
        print(randomAgent.exp, "/", MEMORY_CAPACITY)
    if rand_agent:
        pickle.dump(randomAgent.memory, open( "memory_abs{}.p".format(MEMORY_CAPACITY), "wb" ))
        agent.memory = randomAgent.memory
    # else:
    #     #pass
    #     # agent.memory = randomAgent.memory
    #     agent.memory = pickle.load( open( "memory_abs{}.p".format(MEMORY_CAPACITY), "rb" ) )

randomAgent = None
iteration = 1
env = Environment(viz=False)
viz_flag = False

print("Starting learning")
while iteration < trials:
    reward, update_ep = env.run(agent,viz_flag,ep_id=iteration)
    iteration += 1
    avgreward.append(reward)
    x.append(iteration)
    avg_reward = np.mean(avgreward)
    viz_flag = True if avg_reward > -.4 else False
    y.append(avg_reward)
    points.set_data(x, y)
    if len(avgreward) > 10:
        maxsofar = max(maxsofar,np.mean(avgreward))

    if True:
        # restore background
        plt.pause(0.05)
        fig.canvas.restore_region(background)
        ax.set_xlim(max(iteration-500,0), iteration+100)
        ax.set_ylim(-1, 1)
        rand = plt.plot((0, trials), (-.75, -.75), 'k-')
        max_.remove()
        max_, = ax.plot((0, trials), (maxsofar, maxsofar), 'k-', color = 'g')
        thresh.remove()
        thresh, = ax.plot((0, trials), (notify_value, notify_value), 'k-', color = 'r')
        #upep.remove()
        upep, = ax.plot((update_ep, update_ep), (-1, 1), 'k-', color = 'r')
        # redraw just the points
        ax.draw_artist(points)
        # fill in the axes rectangle
        fig.canvas.blit(ax.bbox)
    else:
        # redraw everything
        fig.canvas.draw()
    if maxsofar >= notify_value and len(avgreward) == 10000:
        title = "{} Reached".format(notify_value)
        notify_value = float(input())
    print("episode: {}, average reward: {}, Reward: {}, Memory: {}/{}, Epsilon: {:.2f}, Max: {:.2f}, Exp: {}".format(iteration,str(np.round(np.mean(avgreward), precision)),str(np.round(reward, precision)), agent.memory.tree.write , MEMORY_CAPACITY, agent.epsilon,maxsofar,agent.exp))

# finally:
#     pass#agent.brain.model.save("Seaquest-DQN-PER.h5")