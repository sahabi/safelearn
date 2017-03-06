import os, pygame, pygame.locals, sys, random, math
import numpy as np
from pybrain.datasets import SupervisedDataSet
from time import sleep
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
        obstacle_x_min = 230
        obstacle_y_min = 130
        obstacle_x_max = 250
        obstacle_y_max = 150
        self.OBSTACLES = [(obstacle_x_min,obstacle_y_min,obstacle_x_max,obstacle_y_max)]
        self.obstaclePixels = [[False for i in range(0,self.YSIZE)] for j in range(0,self.XSIZE)]
        for (a,b,c,d) in self.OBSTACLES:
            for i in range(a,c):
                for j in range(b,d):
                    self.obstaclePixels[i][j] = True
        self.CRASH_COST = 1
        self.GOAL_LINE_REWARD = 1
        self.TRAIN_EVERY_NTH_STEP = 8
        # Prepare screen
        self.screen = pygame.display.set_mode((self.XSIZE,self.YSIZE))
        pygame.display.set_caption('Learning Visualizer')
        self.clock = pygame.time.Clock()
        self.isPaused = False
        self.screenBuffer = pygame.Surface(self.screen.get_size())
        self.screenBuffer = self.screenBuffer.convert()
        self.predictionBuffer = pygame.Surface((self.XSIZE/self.MAGNIFY,self.YSIZE/self.MAGNIFY))
        self.predictionBuffer.fill((64, 64, 64)) # Dark Gray
        pygame.font.init()
        self.usedfont = pygame.font.SysFont("monospace", 15)

        self.currentPos = (100.0,100.0)
        self.currentDir = random.random()*math.pi*2
        self.currentSpeedPerStep = 1.0
        self.currentRotationPerStep = 0.04
        # There are multiple view of the window. Here, we store the current state
        self.displayBufferEmpty = True
        self.isLearning = True
        self.clock = pygame.time.Clock()
        self.usedfont = pygame.font.SysFont("monospace", 15)

        self.allPixelsDS = [[np.array([]) for i in range(0,self.NOFPIXELSPLITS)] for i in xrange(0,8)]
        pixels = []

        for x in range(0,self.XSIZE/self.MAGNIFY):
            for y in range(0,self.YSIZE/self.MAGNIFY):
                if not self.obstaclePixels[x*self.MAGNIFY][y*self.MAGNIFY]:
                    pixels.append((float(x)*self.MAGNIFY/self.XSIZE,float(y)*self.MAGNIFY/self.YSIZE))
        random.shuffle(pixels)


        for i in range(0,self.NOFPIXELSPLITS):
            thisChunk = len(pixels)/(self.NOFPIXELSPLITS-i)
            for j in range(0,thisChunk):
                for k in range(0,8):
                    a = np.array([pixels[j][0],pixels[j][1],math.sin(k*0.25*math.pi),math.cos(k*0.25*math.pi)])
                    #a = np.array([pixels[j][0],pixels[j][1]])
                    b = self.allPixelsDS[k][i]
                    if b.shape[0] == 0:
                        self.allPixelsDS[k][i] = a
                    else:
                        self.allPixelsDS[k][i] = np.vstack((a,b))
            pixels = pixels[thisChunk:]
        self.displayDirection = 0
        self.iteration = 0
        #print self.allPixelsDS
        for x in range(obstacle_x_min,obstacle_x_max):
            for y in range(obstacle_y_min,obstacle_y_max):
                self.screenBuffer.set_at((x,y),(0,0,0))

    def reset(self,net,iteration):
        #self.currentPos = (400.0,400.0)
        self.currentPos = (random.uniform(0,1)*self.XSIZE,random.uniform(0,1)*self.YSIZE)
        self.currentDir = 5
        self.currentSpeedPerStep = 1.0
        self.currentRotationPerStep = 0.04
        self.iteration += 1
#(x,y,direcSin,direcCos)
        if pygame.display.get_active():
            color = (255*0/8.0,0,0)
            allActivations = net.predict_all(self.allPixelsDS[self.displayDirection][iteration % self.NOFPIXELSPLITS])
            for i,(x,y,direcSin,direcCos) in enumerate(self.allPixelsDS[self.displayDirection][iteration % self.NOFPIXELSPLITS]):
            #for i,(x,y) in enumerate(self.allPixelsDS[self.displayDirection][iteration % self.NOFPIXELSPLITS]):      
                arg = argmax(allActivations[i])
                if arg == 7:
                    color = (0,40,0) #dark blue
                if arg == 6:
                    color = (0,122,0) #blue
                if arg == 5:
                    color = (0,255,0) #light blue
                if arg == 4:
                    color = (255,255,255) #white
                elif arg == 3:
                    color = (255,0,0) #light red
                elif arg == 2:
                    color = (122,0,0) #red
                elif arg == 1:
                    color = (40,0,0) #dark red
                elif arg == 0:
                    color = (0,0,0) #black
                self.predictionBuffer.set_at((int(round(x*self.XSIZE/self.MAGNIFY)), int(round(y*self.YSIZE/self.MAGNIFY))), color)
            # Scale up the "predictionBuffer"
            self.screenBuffer.blit(pygame.transform.smoothscale(self.predictionBuffer, (self.XSIZE, self.YSIZE)),(0,0))
            label = self.usedfont.render("Best action, dir: "+str(self.displayDirection)+", learning: ", 1, (255,255,0))
            self.screenBuffer.blit(label, (1, 1))
            self.displayBufferEmpty = False
        else:
            if not self.displayBufferEmpty:
                # When the buffer comes back up, we don't want 
                self.predictionBuffer.fill((64, 64, 64))
                self.displayBufferEmpty = True       

        # ====================================
        # Draw origin of the motion
        pygame.draw.line(self.screenBuffer,(0,0,255),(self.currentPos[0]-2,self.currentPos[1]-2),(self.currentPos[0]+2,self.currentPos[1]+2),3)
        pygame.draw.line(self.screenBuffer,(0,0,255),(self.currentPos[0]+2,self.currentPos[1]-2),(self.currentPos[0]-2,self.currentPos[1]+2),3)  

        for event in pygame.event.get():
                    
            if event.type == pygame.locals.QUIT or (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_ESCAPE):           
                sys.exit(0)
            if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_SPACE):
                self.displayQValue = not self.displayQValue            
            if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_l):
                self.isLearning = not self.isLearning
            if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_RIGHT):
                #print self.displayDirection
                self.displayDirection = (self.displayDirection + 1) % 8
            if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_LEFT):
                self.displayDirection = (self.displayDirection + 7) % 8

        return np.array([self.currentPos[0]/self.XSIZE, self.currentPos[1]/self.YSIZE, math.sin(self.currentDir*0.25*math.pi)\
            ,math.cos(self.currentDir*0.25*math.pi)])       
        #return np.array([self.currentPos[0]/self.XSIZE, self.currentPos[1]/self.YSIZE])   
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
            pygame.draw.line(self.screenBuffer,(0,0,255),self.oldPos,self.currentPos,3)

        if (self.currentPos[0]>=self.XSIZE) or (self.currentPos[0]<0) or (self.currentPos[1]>=self.YSIZE) or (self.currentPos[1]<0):
            self.currentPos = (random.random()*self.XSIZE,random.random()*self.YSIZE)
            self.currentDir = random.random()*math.pi*2
            R = -1*self.CRASH_COST
            done = True
        elif self.obstaclePixels[int(self.currentPos[0])][int(self.currentPos[1])]:
            self.currentPos = (random.random()*self.XSIZE,random.random()*self.YSIZE)
            self.currentDir = random.random()*math.pi*2
            R = -1*self.CRASH_COST
            done = True            
        elif ((self.currentPos[1]>self.YSIZE/2) and (self.currentPos[0]<self.XSIZE/2) and (stepStartingPos[0]>self.XSIZE/2)) or ((self.currentPos[1]<self.YSIZE/2) and (self.currentPos[0]<self.XSIZE/2) and (stepStartingPos[0]>self.XSIZE/2)):
            R = self.GOAL_LINE_REWARD*1
            done = True
        elif ((self.currentPos[1]>self.YSIZE/2) and (self.currentPos[0]>self.XSIZE/2) and (stepStartingPos[0]<self.XSIZE/2)) or ((self.currentPos[1]<self.YSIZE/2) and (self.currentPos[0]>self.XSIZE/2) and (stepStartingPos[0]<self.XSIZE/2)):
            R = self.GOAL_LINE_REWARD*1
            done = True
        else:
            R = 0.00
            done = False

        S_dash = np.array([self.currentPos[0]/self.XSIZE, self.currentPos[1]/self.YSIZE,math.sin(self.currentDir*0.25*math.pi),math.cos(self.currentDir*0.25*math.pi)])
        #S_dash = np.array([self.currentPos[0]/self.XSIZE, self.currentPos[1]/self.YSIZE])
        if pygame.display.get_active():        
            self.screen.blit(self.screenBuffer, (0, 0))
            pygame.display.flip()
        return (S_dash, R, done)

    def close(self):
        sys.exit(0)

