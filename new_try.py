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
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from dialogue import Widget
import random, numpy, math, scipy
from SumTree import SumTree
import env_m


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
        model.compile(lr=.01, optimizer='rmsprop', loss='mse')

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
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

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
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 200000 

BATCH_SIZE = 60

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.05

EXPLORATION_STOP = 500000   # at this step epsilon will be 0.01
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 10

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.exp = 0

        self.brain = Brain(stateCnt, actionCnt)
        # self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)
        self.exp += 1

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

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
    memory = Memory(MEMORY_CAPACITY)
    

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt
        self.brain = Brain(8, actionCnt)
        self.exp = 0

    def act(self, s):
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # reward
        self.memory.add(error, sample)
        self.exp += 1

    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self):
        self.env = env_m.Env()

    def run(self, agent, iteration=0, viz=False):          
        s = self.env.reset(agent.brain,iteration,viz=viz)
        # w = processImage(img)
        # s = numpy.array([w, w])

        R = 0
        while True:         
            a = agent.act(s)

            r = 0
            s_, r, done = self.env.step(a)

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )        

            s = s_
            R += r
            iteration += 1

            if done:
                agent.replay() 
                break

        #print("Total reward:", R)
        return R

#-------------------- MAIN ----------------------------
env = Environment()

stateCnt  = (8, 1)
actionCnt = 8

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

notify_value = -1
t = 0
avgreward = deque([],100)
trials = 10000
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

plt.show(False)
plt.draw()
maxsofar = -2
max_, = ax.plot((0, 5000), (maxsofar, maxsofar), 'k-')
thresh, = ax.plot((0, 5000), (notify_value, notify_value), 'k-') 
if True:
    # cache the background
    background = fig.canvas.copy_from_bbox(ax.bbox)

points = ax.plot(x, y, 'o')[0]
rand = 10000
try:
    print("Initialization with random agent...")
    while randomAgent.exp < rand:
        env.run(randomAgent)
        print(randomAgent.exp, "/", rand)

    agent.memory = randomAgent.memory

    randomAgent = None
    iteration = 1


    print("Starting learning")
    while True:
        if maxsofar >= -.5:
            reward = env.run(agent,iteration,viz=True)
        else:
            reward = env.run(agent,iteration)
        iteration += 1
        avgreward.append(reward)
        x.append(iteration)
        y.append(np.mean(avgreward))
        points.set_data(x, y)
        if len(avgreward) > 10:
            maxsofar = max(maxsofar,np.mean(avgreward))

        if True:
            # restore background
            plt.pause(0.05)
            fig.canvas.restore_region(background)
            ax.set_xlim(max(iteration-500,0), iteration+100)
            ax.set_ylim(-2, 2)
            rand = plt.plot((0, trials), (-.75, -.75), 'k-')
            max_.remove()
            max_, = ax.plot((0, trials), (maxsofar, maxsofar), 'k-', color = 'g')
            thresh.remove()
            thresh, = ax.plot((0, trials), (notify_value, notify_value), 'k-', color = 'r')
            # redraw just the points
            ax.draw_artist(points)
            # fill in the axes rectangle
            fig.canvas.blit(ax.bbox)
        else:
            # redraw everything
            fig.canvas.draw()
        if maxsofar >= notify_value and len(avgreward) == 100:
            title = "{} Reached".format(notify_value)
            # w = Widget(notify_value)
            # w.root.mainloop()
            notify_value = float(input())
        print("episode: {}, average reward: {}, Reward: {}, Memory: {}/{}, Epsilon: {}, Max: {}".format(iteration,np.mean(avgreward),reward, agent.exp, 1, agent.epsilon,maxsofar))

finally:
    pass#agent.brain.model.save("Seaquest-DQN-PER.h5")