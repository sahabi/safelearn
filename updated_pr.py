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
from agent_pr import agent
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from dialogue import Widget

LEFT = 0
RIGHT = 1
MAX_TIMESTEPS = 500

blob = agent(4,[i for i in range(0,8)], epsilon=1)
env = env_m.Env()
#env = wrappers.Monitor(env, '/tmp/cartpole-experiment-v1',force=True)
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

for i_episode in range(trials):
    
    S = env.reset(blob.Q_est, t)
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
    if i_episode > 20:
        blob.reflect(i_episode)
        
    # when the episode ends the agent will have hit a terminal state so give it a zero reward
    if t < MAX_TIMESTEPS:
        blob.observe(S,A,0.,None)
    else:
        blob.observe(S,A,1.,None)
            
    avgreward.append(tot_R)
        # update the xy data
    x.append(i_episode)
    y.append(np.mean(avgreward))
    points.set_data(x, y)
    if len(avgreward) > 10:
        maxsofar = max(maxsofar,np.mean(avgreward))

    if True:
        # restore background
        plt.pause(0.05)
        fig.canvas.restore_region(background)
        ax.set_xlim(max(i_episode-500,0), i_episode+100)
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
    print("episode: {}, average reward: {}, Reward: {}, Memory: {}/{}, Epsilon: {}, Max: {}".format(i_episode,np.mean(avgreward),tot_R, len(blob.experience_pr._experience), 1, blob.policy.epsilon,maxsofar))
plt.close(fig)
env.close()