import os, pygame, pygame.locals, sys, random, math
import numpy as np
from pybrain.datasets import SupervisedDataSet

# Helper functions
def argmax(b):
    maxVal = None
    maxData = None
    for i,a in enumerate(b):
        if a>maxVal:
            maxVal = a
            maxData = i
    return maxData

class Env(object):
    def __init__(self):
        # CONSTANTS for how large the drawing window is.
        self.XSIZE = 480
        self.YSIZE = 480
        # When visualizing the learned policy, in order to speed up things, we only a fraction of all pixels on a lower resolution. Here are the parameters for that.
        self.MAGNIFY = 2
        self.NOFPIXELSPLITS = 16
        # Obstacle definitions
        self.OBSTACLES = [(230,100,250,350)]
        self.obstaclePixels = [[False for i in range(0,YSIZE)] for j in range(0,XSIZE)]
        for (a,b,c,d) in self.OBSTACLES:
            for i in range(a,c):
                for j in range(b,d):
                    self.obstaclePixels[i][j] = True
        self.CRASH_COST = 1
        self.GOAL_LINE_REWARD = 1
        self.TRAIN_EVERY_NTH_STEP = 10
        # Prepare screen
        self.screen = pygame.display.set_mode((XSIZE,YSIZE))
        pygame.display.set_caption('Learning Visualizer')
        self.clock = pygame.time.Clock()
        self.isPaused = False
        self.screenBuffer = pygame.Surface(screen.get_size())
        self.screenBuffer = screenBuffer.convert()
        self.predictionBuffer = pygame.Surface((XSIZE/MAGNIFY,YSIZE/MAGNIFY))
        self.predictionBuffer.fill((64, 64, 64)) # Dark Gray
        self.pygame.font.init()
        self.usedfont = pygame.font.SysFont("monospace", 15)

        self.currentPos = (100.0,100.0)
        self.currentDir = 0.0
        self.currentSpeedPerStep = 1.0
        self.currentRotationPerStep = 0.04
        # There are multiple view of the window. Here, we store the current state
        displayBufferEmpty = True
        isLearning = True
        usedfont = pygame.font.SysFont("monospace", 15)
        displayDirection = 0

    def reset(self):
        self.currentPos = (100.0,100.0)
        self.currentDir = 0.0
        self.currentSpeedPerStep = 1.0
        self.currentRotationPerStep = 0.04
        return np.array([self.currentPos[0]/self.XSIZE, self.currentPos[1]/self.YSIZE, math.sin(self.currentDir*0.25*math.pi)\
            ,math.cos(self.currentDir*0.25*math.pi)])       

    def step(self, action):

        targetDirDiscrete = action
        targetDir = targetDirDiscrete*math.pi*2/8.0
        stepStartingPos = self.currentPos
        
        # Simulate the cars for some steps. Also draw the trajectory of the car.
        for i in range(0,self.TRAIN_EVERY_NTH_STEP):
            if (self.currentDir>math.pi*2):
                self.currentDir -= 2*math.pi
            if targetDir < self.currentDir:
                self.currentDir = max(targetDir,self.currentDir-self.currentRotationPerStep)
            else:
                self.currentDir = min(targetDir,self.currentDir+self.currentRotationPerStep)
            self.oldPos = self.currentPos
            self.currentPos = (self.currentPos[0]+self.currentSpeedPerStep*math.sin(self.currentDir),self.currentPos[1]+self.currentSpeedPerStep*math.cos(self.currentDir))
            pygame.draw.line(self.screenBuffer,(0,255,255),self.oldPos,self.currentPos)
        print currentPos[0]
        print currentPos[1]
        #print stepStartingPos[0]
        if (self.currentPos[0]>XSIZE) or (self.currentPos[0]<0) or (self.currentPos[1]>YSIZE) or (self.currentPos[1]<0) or ((self.currentPos[1]>YSIZE/2) and (self.currentPos[0]>XSIZE/2) and (stepStartingPos[0]<XSIZE/2)):
            self.currentPos = (random.random()*XSIZE,random.random()*YSIZE)
            self.currentDir = random.random()*math.pi*2
            nofStepsLeft = 0
            R = -1*self.CRASH_COST
            done = True
        elif (self.currentPos[1]>self.YSIZE/2) and (self.currentPos[0]<self.XSIZE/2) and (self.stepStartingPos[0]>self.XSIZE/2):
            R = self.GOAL_LINE_REWARD
            done = True
        else:
            R = 0.0

        S_dash = np.array([self.currentPos[0]/self.XSIZE, self.currentPos[1]/self.YSIZE,math.sin(self.currentDir*0.25*math.pi),math.cos(self.currentDir*0.25*math.pi)])
        return (S_dash, R, done)

    def close(self):
        sys.exit(0)
