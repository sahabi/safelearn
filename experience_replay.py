# Deep Q-learning requires that the training data be randomly selected and fed into the network to avoid
# the so called catastrophic forgetting problem. The experienceReplay class does this by storing instances of 
# memoryNode in a big double ended queue and returning a random selection of them on calls to 'recall'

# The memoryNode class stores the state, action, reward info as well as pointing to the next node from that experience.
# The let's me get the next state but later on it'll help me implement an n-step return to try to improve my returns estimates

from collections import deque
import random as rnd
import numpy as np

def argmax(b):
    maxVal = None
    maxData = None
    for i,a in enumerate(b):
        if a>maxVal:
            maxVal = a
            maxData = i
    return maxData

class memoryNode:
    def __init__(self, S = None, A = None, R = None, nextNode = None, TD = 1.0):
        self.S = S
        self.A = A
        self.R = R
        self.TD = TD
        self.next = nextNode

    def __str__(self):
        return "S: {} A: {} R: {}" .format(self.S,self.A,self.R)
        
    def stepAhead(self, n):
        node = self
        while n > 0 and node.next != None:
            yield node
            n = n - 1
            node = node.next

class experienceReplay:
    
    def __init__(self,bufferSize):
        self.buffer = deque([],bufferSize)
        self.bufferSize = bufferSize
        self.sum_p = 0

    def recall(self, Q, Q_est, batchSize=32):
        j = 0
        TDs = [(y,x.TD) if (type(x.TD)==float or type(x.TD)==int  or type(x.TD)==np.float64)  else (y,x.TD[0]) for y,x in zip(range(0,len(self.buffer)-1),self.buffer)]
        TDs.sort(key=lambda x: x[1], reverse=True)
        #idx = range(0,len(self.buffer)-1)
        idx = [x[0] for x in TDs]
        idx = range(0,len(self.buffer)-1)
        k = min(batchSize, len(self.buffer))
        end = 0
        # for i in range(k):
        #     acc = 0
        #     for q,z in enumerate(idx):
        #         acc += self.buffer[z].TD
        #         if acc > self.sum_p/float(k):
        #             gap = q
        #     start = end
        #     end = start + gap
        #     partitions[i] = (start, end)
        # for start, end in partitions:
        #     new_idx.appendrnd.choice(idx[start:end])

        rnd.shuffle(idx)
        # print partitions
        # for every shuffled id
        for jj,i in enumerate(idx):
            
            # if the corresponding element has a nextState then yield it
            if self.buffer[i].next != None and self.buffer[idx[jj-1]].S != None:  
                if len((self.buffer[i].R + 1 * \
                Q_est.predict(self.buffer[i].S[np.newaxis])[0][argmax(Q.predict(self.buffer[i].S[np.newaxis])[0])] - \
                Q.predict(self.buffer[idx[jj-1]].S[np.newaxis])[0][self.buffer[idx[jj-1]].A]).shape)>0:
                    self.buffer[i].TD = abs(self.buffer[i].R + 1 * \
                Q_est.predict(self.buffer[i].S[np.newaxis])[0][argmax(Q.predict(self.buffer[i].S[np.newaxis])[0])] - \
                Q.predict(self.buffer[idx[jj-1]].S[np.newaxis])[0][self.buffer[idx[jj-1]].A][0])
                    print("!!!!!!!!!!")
                else:
                    self.buffer[i].TD = abs(self.buffer[i].R + 1 * \
                Q_est.predict(self.buffer[i].S[np.newaxis])[0][argmax(Q.predict(self.buffer[i].S[np.newaxis])[0])] - \
                Q.predict(self.buffer[idx[jj-1]].S[np.newaxis])[0][self.buffer[idx[jj-1]].A])
                #self.buffer[i].TD = 1
                yield self.buffer[i]
                j += 1
                
            # and if we've returned enough samples then break
            if j == batchSize:
                break
        
    def remember(self, state, action, reward, nextState):
        # if the buffer is empty or the last state was terminal then we need to start afresh
        if len(self.buffer) == 0 or self.buffer[-1].S == None:
            self.buffer.append(memoryNode(state))
            
        memory = memoryNode(nextState)
        
        # fill in the remaining details for the current state
        self.buffer[-1].A = action
        self.buffer[-1].R = reward
        # print [type(x.TD) for x in self.buffer]
        self.buffer[-1].TD = max([x.TD if (type(x.TD)==float or type(x.TD)==int or type(x.TD)==np.float64) else x.TD[0] for x in self.buffer])
        self.buffer[-1].next = memory
        self.sum_p = sum([x.TD  if (type(x.TD)==float or type(x.TD)==int or type(x.TD)==np.float64)  else x.TD[0] for x in self.buffer])
        
        # append a new memory for the current state
        self.buffer.append(memory)
