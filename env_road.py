import os, pygame, pygame.locals, sys, random, math
import numpy as np
#from pybrain.datasets import SupervisedDataSet
from time import sleep
# Helper functions


class obstacle(object):
    def __init__(self,xmin,xmax,ymin,ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

def argmax(b):
    maxVal = -100000000
    maxData = None
    for i,a in enumerate(b):
        if a>maxVal:
            maxVal = a
            maxData = i
    return maxData

def make_horizontal_wall(obs_left, obs_right,loc='top'):
    if loc == 'bot':
        return obstacle(obs_left.xmin,obs_right.xmax,min(obs_left.ymax,obs_right.ymax),min(obs_left.ymax,obs_right.ymax)+20)
    if loc == 'top':
        return obstacle(obs_left.xmin,obs_right.xmax,max(obs_left.ymin,obs_right.ymin)-20,max(obs_left.ymin,obs_right.ymin))

class Env(object):
    def __init__(self,viz=True):
        # CONSTANTS for how large the drawing window is.
        self.XSIZE = 480
        self.YSIZE = 480
        # When visualizing the learned policy, in order to speed up things, we only a fraction of all pixels on a lower resolution. Here are the parameters for that.
        self.MAGNIFY = 2
        self.NOFPIXELSPLITS = 4
        self.viz = viz
        # Obstacle definitions
        obs1 = obstacle(130,150,120,420)
        obs2 = obstacle(40,60,40,420)
        obs6 = obstacle(430,450,20,420)
        obs7 = obstacle(330,350,120,350)
        obs8 = obstacle(240,260,350,440)
        obs4 = make_horizontal_wall(obs2,obs6,'top')
        obs3 = make_horizontal_wall(obs2,obs1,'bot')
        obs5 = make_horizontal_wall(obs1,obs7,'top')
        obs10 = make_horizontal_wall(obs8,obs6,'bot')
        obs9 = obstacle(obs8.xmin,obs7.xmax,obs7.ymax,obs7.ymax+20)
        
        self.OBSTACLES = [obs1,obs2,obs3,obs4,obs5,obs6,obs7,obs8,obs9,obs10]
        self.obstaclePixels = [[False for i in range(0,self.YSIZE)] for j in range(0,self.XSIZE)]
        for obs in self.OBSTACLES:
            for i in range(obs.xmin,obs.xmax):
                for j in range(obs.ymin,obs.ymax):
                    self.obstaclePixels[i][j] = True
        self.CRASH_COST = 1
        self.GOAL_LINE_REWARD = 1
        self.TRAIN_EVERY_NTH_STEP = 6
        self.currentPos = (100.0,100.0)
        self.currentDir = random.random()*math.pi*2
        self.currentSpeedPerStep = 1.0
        self.currentRotationPerStep = 0.04
        # There are multiple view of the window. Here, we store the current state
        self.displayBufferEmpty = True
        self.isLearning = True
        # Prepare screen
        #if self.viz:
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
        self.clock = pygame.time.Clock()
        self.usedfont = pygame.font.SysFont("monospace", 15)          
        self.allPixelsDS = [[np.array([]) for i in range(0,self.NOFPIXELSPLITS)] for i in range(0,8)]
        pixels = []

        for x in range(0,int(self.XSIZE/self.MAGNIFY)):
            for y in range(0,int(self.YSIZE/self.MAGNIFY)):
                if not self.obstaclePixels[x*self.MAGNIFY][y*self.MAGNIFY]:
                    pixels.append((float(x)*self.MAGNIFY/self.XSIZE,float(y)*self.MAGNIFY/self.YSIZE))
        random.shuffle(pixels)

        for i in range(0,self.NOFPIXELSPLITS):
            thisChunk = int(len(pixels)/(self.NOFPIXELSPLITS-i))
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
        for obs in self.OBSTACLES:
            for x in range(obs.xmin,obs.xmax):
                for y in range(obs.ymin,obs.ymax):
                    self.screenBuffer.set_at((x,y),(0,0,0))

        self.displayDirection = 0
        self.iteration = 0
        #print self.allPixelsDS

    def inr1(self,x,y):
        if self.OBSTACLES[1].xmax <= x <= self.OBSTACLES[0].xmin and self.OBSTACLES[1].ymin <= y <= self.OBSTACLES[1].ymax:
            return True

    def inr2(self,x,y):
        if self.OBSTACLES[3].xmin <= x <= self.OBSTACLES[3].xmax and self.OBSTACLES[3].ymax <= y <= self.OBSTACLES[4].ymin :
            return True

    def inr3(self,x,y):
        if self.OBSTACLES[6].xmax <= x <= self.OBSTACLES[5].xmin and self.OBSTACLES[5].ymin <= y <= self.OBSTACLES[5].ymax:
            return True

    def inr4(self,x,y):
        if self.OBSTACLES[9].xmin <= x <= self.OBSTACLES[9].xmax and self.OBSTACLES[8].ymax <= y <= self.OBSTACLES[9].ymin:
            return True

    def inside(self,x,y):
        #R1
        if self.inr1(x,y):
            return True
        #R2
        elif self.inr2(x,y):
            return True
        #R3
        elif self.inr3(x,y):
            return True
        #R4
        elif self.inr4(x,y): 
            return True
        else:
            return False

    def reset(self,net,iteration,viz=True):
        #self.currentPos = (400.0,400.0)
        rand_x = random.random()*self.XSIZE
        rand_y = random.random()*self.YSIZE
        while not self.inside(rand_x,rand_y):
            rand_x = random.random()*self.XSIZE
            rand_y = random.random()*self.YSIZE     
        self.currentPos = (.25*self.XSIZE,.1*self.YSIZE)
        rand_x = 100
        rand_y = 400
        self.currentDir = math.pi
        #self.currentPos = (.25*self.XSIZE,.1*self.YSIZE)
        self.currentPos = (rand_x, rand_y)
        self.currentSpeedPerStep = 1.0
        self.currentRotationPerStep = 0.04
        self.iteration += 1
#(x,y,direcSin,direcCos)
        self.viz = viz
        if pygame.display.get_active() and self.viz:
            color = (255*0/8.0,0,0)
            a = np.reshape(self.allPixelsDS[self.displayDirection][iteration % self.NOFPIXELSPLITS],(self.allPixelsDS[self.displayDirection][iteration % self.NOFPIXELSPLITS].shape[0],1,4))
            allActivations = net.predict_all(a)
            for i,(x,y,direcSin,direcCos) in enumerate(self.allPixelsDS[self.displayDirection][iteration % self.NOFPIXELSPLITS]):
            #for i,(x,y) in enumerate(self.allPixelsDS[self.displayDirection][iteration % self.NOFPIXELSPLITS]):      
                arg = argmax(allActivations[i])
                if arg == 7:
                    color = (0,40,0) #dark green
                if arg == 6:
                    color = (0,122,0) #green
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
        if self.viz:
            pygame.draw.line(self.screenBuffer,(0,0,255),(self.currentPos[0]-2,self.currentPos[1]-2),(self.currentPos[0]+2,self.currentPos[1]+2),3)
            pygame.draw.line(self.screenBuffer,(0,0,255),(self.currentPos[0]+2,self.currentPos[1]-2),(self.currentPos[0]-2,self.currentPos[1]+2),3)  

        if self.viz:
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
            elif (self.currentDir<0):
                self.currentDir += 2*math.pi
            if targetDir < self.currentDir:
                if ((2*math.pi - self.currentDir) + targetDir) > (self.currentDir - targetDir):
                    self.currentDir = max(targetDir,self.currentDir-self.currentRotationPerStep)
                else:
                    self.currentDir = max(targetDir,self.currentDir+self.currentRotationPerStep)
            else:
                if ((2*math.pi - targetDir) + self.currentDir) > (targetDir - self.currentDir):
                    self.currentDir = min(targetDir,self.currentDir+self.currentRotationPerStep)
                else:
                    self.currentDir = min(targetDir,self.currentDir-self.currentRotationPerStep)

            self.oldPos = self.currentPos
            self.currentPos = (self.currentPos[0]+self.currentSpeedPerStep*math.sin(self.currentDir),self.currentPos[1]+self.currentSpeedPerStep*math.cos(self.currentDir))
            
            if self.viz:
                pygame.draw.line(self.screenBuffer,(0,0,255),self.oldPos,self.currentPos,3)
        # hitting the border
        bad = -1*self.CRASH_COST
        good = self.GOAL_LINE_REWARD*1
        #print inside(self.currentPos[0],self.currentPos[1])
        R = 0

        if not self.inside(self.currentPos[0],self.currentPos[1]):
            done = True
            R += 0

        elif((self.currentPos[1]<self.YSIZE/2) and (self.currentPos[0]>self.XSIZE/2) and (stepStartingPos[0]<self.XSIZE/2)):
            R += 0
            print('checkpoint 1')
            #R = 0
            done = False

        elif ((self.currentPos[1]>self.YSIZE/2) and (self.currentPos[0]>self.XSIZE/2) and (stepStartingPos[1]<self.YSIZE/2)):
            R += 0
            print('checkpoint 2')
            #R = 0
            done = False

        elif ((self.currentPos[1]>self.YSIZE/2) and (self.currentPos[0]<self.XSIZE/2) and (stepStartingPos[0]>self.XSIZE/2)):
            R += 0
            done = True
            print('checkpoint 3')
        else:
            if self.inr1(self.currentPos[0],self.currentPos[1]):
                R += .5*(stepStartingPos[1] - self.currentPos[1])/ self.YSIZE

            if self.inr2(self.currentPos[0],self.currentPos[1]):
                R += .7*(self.currentPos[0] - stepStartingPos[0])/ self.XSIZE

            if self.inr3(self.currentPos[0],self.currentPos[1]):
                R += (self.currentPos[1] - stepStartingPos[1])/ self.YSIZE

            if self.inr4(self.currentPos[0],self.currentPos[1]):
                R += 1.5*(stepStartingPos[0] - self.currentPos[0])/ self.XSIZE
            done = False

        S_dash = np.array([self.currentPos[0]/self.XSIZE, self.currentPos[1]/self.YSIZE,math.sin(self.currentDir*0.25*math.pi),math.cos(self.currentDir*0.25*math.pi)])
        if pygame.display.get_active():        
            self.screen.blit(self.screenBuffer, (0, 0))
            pygame.display.flip()
        return (S_dash, R, done)

    def close(self):
        sys.exit(0)

