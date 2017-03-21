# This algorith is based around DQN, parameterised like so:

# Q-function is a dense network with 2x100 node hidden layers
# experience replay contained the most recent 10000 state, action, reward triplets.
# learning took place after every episode using a minibatch size of 100
# learning rate = 0.01
# gamma = 0.99 
# eGreediness = 0.05

from collections import deque
import env_road as env_m
#from gym import wrappers
import numpy as np
from agent_pr import agent
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import random
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
LEFT = 0
RIGHT = 1
MAX_TIMESTEPS = 500

blob = agent(4,[i for i in range(0,8)], epsilon=0)
model = Sequential()
model.add(Flatten(input_shape=(1,) + (4,)))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('linear'))
model.load_weights('Car_RL_weights.h5f')  # updates V, L and mu model since the weights are shared
blob.Q_est.model = model
blob.Q.model = model
env = env_m.Env()
#env = wrappers.Monitor(env, '/tmp/cartpole-experiment-v1',force=True)
notify_value = -1
t = 0
avgreward = deque([],100)
avgQ = deque([],100)
trials = 1000000
fig, ax = plt.subplots(1, 1)
ax.set_aspect('auto')
ax.set_xlim(0, 5000)
ax.set_ylim(-0.5, 1)
ax.set_ylabel('Rewards')
ax.set_xlabel('Episodes')
ax.hold(True)
x = deque([],500)
x.append(0)
y = deque([],500)
y.append(-1)
xQ = deque([],500)
xQ.append(0)
yQ = deque([],500)
yQ.append(-1)
plt.show(False)
plt.draw()
maxsofar = -2
max_, = ax.plot((0, 5000), (maxsofar, maxsofar), 'k-')
thresh, = ax.plot((0, 5000), (notify_value, notify_value), 'k-') 
if True:
    # cache the background
    background = fig.canvas.copy_from_bbox(ax.bbox)

points = ax.plot(x, y, '-')[0]
pointsQ = ax.plot(xQ, yQ, '-', color='g')[0]
viz_flag = True
S_list = []
q_est_trials = 1000
# for i_episode in range(q_est_trials):
#     print('{}/{}'.format(i_episode,q_est_trials))
#     S = env.reset(blob.Q_est, t, viz_flag)
#     done = False   
#     t = 0
#     tot_R = 0  
#     while not done:
#         t += 1
#         S_list.append(S)
#         A = random.choice([0,1,2,3,4,5,6,7])#blob.act(S)
#         S_dash, R, done = env.step(A)
#         blob.observe(S,A,R,S_dash)
#         #self.Q.predict(state[np.newaxis,:])
#         tot_R += R
#         S = np.copy(S_dash)    

for i_episode in range(trials):
    
    S = env.reset(blob.Q_est, t, viz_flag)
    done = False   
    t = 0
    tot_R = 0  
    while not done:
        t += 1
        #S = np.reshape(S,(1,1,4))
        A = blob.act(S)
        S_dash, R, done = env.step(A)
        #blob.observe(S,A,R,S_dash)
        tot_R += R
        S = np.copy(S_dash)
        
    # every now and then stop, and think things through:
    #if i_episode > 55:
    #    blob.reflect(i_episode)
        
    # when the episode ends the agent will have hit a terminal state so give it a zero reward
    # if t < MAX_TIMESTEPS:
    #     blob.observe(S,A,0.,None)
    # else:
    #     blob.observe(S,A,1.,None)
            
    avgreward.append(tot_R)
    avg_Q = 0
    avgQ.append(avg_Q)
    avg_reward = np.mean(avgreward)
    viz_flag = True if avg_reward > -10 else False
    # update the xy data
    yQ.append(np.mean(avgQ))
    x.append(i_episode)
    y.append(avg_reward)
    points.set_data(x, y)
    pointsQ.set_data(x, yQ)
    if len(avgreward) > 10:
        maxsofar = max(maxsofar,np.mean(avgreward))

    if False:
        # restore background
        plt.pause(0.05)
        fig.canvas.restore_region(background)
        ax.set_xlim(max(i_episode-500,0), i_episode+100)
        ax.set_ylim(-.5, 1)
        rand = plt.plot((0, trials), (-.75, -.75), 'k-')
        max_.remove()
        max_, = ax.plot((0, trials), (maxsofar, maxsofar), 'k-', color = 'g')
        thresh.remove()
        thresh, = ax.plot((0, trials), (notify_value, notify_value), 'k-', color = 'r')
        # redraw just the points
        ax.draw_artist(points)
        ax.draw_artist(pointsQ)
        # fill in the axes rectangle
        fig.canvas.blit(ax.bbox)
    else:
        # redraw everything
        fig.canvas.draw()
    if maxsofar >= notify_value and len(avgreward) == 100:
        title = "{} Reached".format(notify_value)
        notify_value = float(input())
    #print(np.average(np.amax(blob.Q.model.predict(np.array(S_list)), axis=1)))

    print("episode: {}, average reward: {}, Reward: {:.2f}, Memory: {}/{}, Epsilon: {:.2f}, Max: {:.2f}, Q: {:.2f}".format(i_episode,str(np.round(np.mean(avgreward),3)),tot_R, len(blob.experience_pr._experience), 1, blob.policy.epsilon,maxsofar,np.mean(avgQ)))
plt.close(fig)
env.close()