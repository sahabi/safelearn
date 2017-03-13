#!/usr/bin/env python2
#
# Testing Reinforcement learning with PyBrain
#
import os, pygame, pygame.locals, sys, random, math

# CONSTANTS for how large the drawing window is.
XSIZE = 480
YSIZE = 480

# When visualizing the learned policy, in order to speed up things, we only a fraction of all pixels on a lower resolution. Here are the parameters for that.
MAGNIFY = 2
NOFPIXELSPLITS = 16

# Obstacle definitions
OBSTACLES = [(230,100,250,350)]
obstaclePixels = [[False for i in range(0,YSIZE)] for j in range(0,XSIZE)]
for (a,b,c,d) in OBSTACLES:
    for i in range(a,c):
        for j in range(b,d):
            obstaclePixels[i][j] = True

# Learning parameters
CRASH_COST = 1
GOAL_LINE_REWARD = 1
ALPHA_FACTOR = 0.9
TRAIN_EVERY_NTH_STEP = 10

# Helper functions
def argmax(b):
    maxVal = None
    maxData = None
    for i,a in enumerate(b):
        if a>maxVal:
            maxVal = a
            maxData = i
    return maxData

# Let's build a neural network!
# The network gets as input the current (x,y) coordinates and the sine and cosine of the current heading. Output is one of eight directions towards which the car should rotate to.
#
from pybrain.tools.shortcuts import buildNetwork
from pybrain import SoftmaxLayer
from pybrain import LinearLayer
from pybrain import TanhLayer
net = buildNetwork(4, 25, 25, 8, hiddenclass=TanhLayer, outclass=LinearLayer, bias=True) # outclass=SoftmaxLayer

# The data set for learning - we will later add things
from pybrain.datasets import SupervisedDataSet
ds = SupervisedDataSet(4, 8)

# ==========================================================
# To help with speeding up painting, the pixels in the image are split
# into 16 groups, and in each frame, only the pixels in one group is
# used. We compute the lists of pixels here.
# ==========================================================
pixels = []
for x in range(0,XSIZE/MAGNIFY):
    for y in range(0,YSIZE/MAGNIFY):
        if not obstaclePixels[x*MAGNIFY][y*MAGNIFY]:
            pixels.append((float(x)*MAGNIFY/XSIZE,float(y)*MAGNIFY/YSIZE))
random.shuffle(pixels)

allPixelsDS = [[SupervisedDataSet(4, 8) for i in range(0,NOFPIXELSPLITS)] for i in xrange(0,8)]
for i in range(0,NOFPIXELSPLITS):
    thisChunk = len(pixels)/(NOFPIXELSPLITS-i)
    for j in range(0,thisChunk):
        for k in range(0,8):
            allPixelsDS[k][i].addSample((pixels[j][0],pixels[j][1],math.sin(k*0.25*math.pi),math.cos(k*0.25*math.pi)),(0,0,0,0,0,0,0,0))
    pixels = pixels[thisChunk:]


# Initialize the "Trainer" for the learner.
from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds, momentum=0.0, learningrate=0.03, lrdecay=0.999999, verbose=True, batchlearning=False, weightdecay=0.0000)
startingAlphaValueOfTrainer = trainer.descent.alpha

# Prepare screen
screen = pygame.display.set_mode((XSIZE,YSIZE))
pygame.display.set_caption('Learning Visualizer')
clock = pygame.time.Clock()
isPaused = False
screenBuffer = pygame.Surface(screen.get_size())
screenBuffer = screenBuffer.convert()
predictionBuffer = pygame.Surface((XSIZE/MAGNIFY,YSIZE/MAGNIFY))
predictionBuffer.fill((64, 64, 64)) # Dark Gray
pygame.font.init()
usedfont = pygame.font.SysFont("monospace", 15)

# Initial state of the car
iteration = 0
currentPos = (100.0,100.0)
currentDir = 0.0
currentSpeedPerStep = 1.0
currentRotationPerStep = 0.04

# There are multiple view of the window. Here, we store the current state
displayQValue = True # Alternative: Display best mode
displayDirection = 0
displayBufferEmpty = True
isLearning = True

# Main loop
while 1:

    iteration += 1

    # ==============================================================
    # Painting part one: Draw a visualization of the value function
    # *or* the currently best actions.
    # ==============================================================
    if pygame.display.get_active():
        for i,((x,y,direcSin,direcCos),j) in enumerate(allPixelsDS[displayDirection][iteration % NOFPIXELSPLITS]):
            color = (255*0/8.0,0,0)
            predictionBuffer.set_at((int(round(x*XSIZE/MAGNIFY)), int(round(y*YSIZE/MAGNIFY))), color)  
        # Scale up the "predictionBuffer"
        screenBuffer.blit(pygame.transform.smoothscale(predictionBuffer, (XSIZE, YSIZE)),(0,0))
        if displayQValue:
            label = usedfont.render("Q-Value, dir: "+str(displayDirection)+", learning: "+str(isLearning), 1, (255,255,0))
        else:
            label = usedfont.render("Best action, dir: "+str(displayDirection)+", learning: "+str(isLearning), 1, (255,255,0))
        screenBuffer.blit(label, (1, 1))
        displayBufferEmpty = False
    else:
        if not displayBufferEmpty:
            # When the buffer comes back up, we don't want 
            predictionBuffer.fill((64, 64, 64))
            displayBufferEmpty = True       
    # ====================================
    # Now simulate the car
    # ====================================
    nofStepsLeft = 200
    # Draw origin of the motion
    pygame.draw.line(screenBuffer,(0,255,255),(currentPos[0]-2,currentPos[1]-2),(currentPos[0]+2,currentPos[1]+2))
    pygame.draw.line(screenBuffer,(0,255,255),(currentPos[0]+2,currentPos[1]-2),(currentPos[0]-2,currentPos[1]+2))    
    while nofStepsLeft > 0:
        nofStepsLeft -= 1
        
        # Is the next step that the car does random or not? For Q-Learning, the fraction of
        # steps in which the car behaves randomly should start high, but converge to 0.
        isRandomStep = random.random()<trainer.descent.alpha/startingAlphaValueOfTrainer*0.8
        stepStartingPos = currentPos
        oldDir = currentDir
        
        # Find out the Q values of all available actions
        qValues = net.activate((currentPos[0]/XSIZE, currentPos[1]/YSIZE,math.sin(currentDir*0.25*math.pi),math.cos(currentDir*0.25*math.pi)))
        qValues = [min(a,GOAL_LINE_REWARD) for a in qValues]
        qValues = [max(a,-1*CRASH_COST) for a in qValues]
        
        # Now select the action to be taken.
        if isRandomStep:
            targetDirDiscrete = int(random.random()*7.999999)
        else:
            targetDirDiscrete = argmax(qValues)
        targetDir = targetDirDiscrete*math.pi*2/8.0
        
        # Simulate the cars for some steps. Also draw the trajectory of the car.
        for i in range(0,TRAIN_EVERY_NTH_STEP):
            if (currentDir>math.pi*2):
                currentDir -= 2*math.pi
            if targetDir < currentDir:
                currentDir = max(targetDir,currentDir-currentRotationPerStep)
            else:
                currentDir = min(targetDir,currentDir+currentRotationPerStep)
            oldPos = currentPos
            currentPos = (currentPos[0]+currentSpeedPerStep*math.sin(currentDir),currentPos[1]+currentSpeedPerStep*math.cos(currentDir))
            pygame.draw.line(screenBuffer,(0,255,255),oldPos,currentPos)
        
        # Now detect collisions. Collision possibilities:
        # 1. With the boundaries
        # 2. With the obstacles
        # 3. Crossing the finish line in the wrong direction 

        #print obstaclePixels[int(currentPos[0])][int(currentPos[1])]
        if (currentPos[0]>XSIZE) or (currentPos[0]<0) or (currentPos[1]>YSIZE) or (currentPos[1]<0) or ((currentPos[1]>YSIZE/2) and (currentPos[0]>XSIZE/2) and (stepStartingPos[0]<XSIZE/2)):
            #currentPos = (random.random()*XSIZE,random.random()*YSIZE)
            #currentDir = random.random()*math.pi*2
            currentPos = (400.0,400.0)#(random.random()*self.XSIZE,random.random()*self.YSIZE)
            currentDir = 0.0#currentDir = random.random()*math.pi*2
            nofStepsLeft = 0
            rewardTransition = -1*CRASH_COST
        elif obstaclePixels[int(currentPos[0])][int(currentPos[1])]:
            #currentPos = (random.random()*XSIZE,random.random()*YSIZE)
            #currentDir = random.random()*math.pi*2
            currentPos = (400.0,400.0)#(random.random()*self.XSIZE,random.random()*self.YSIZE)
            currentDir = 0.0#currentDir = random.random()*math.pi*2
            nofStepsLeft = 0
            rewardTransition = -1*CRASH_COST            
        elif (currentPos[1]>YSIZE/2) and (currentPos[0]<XSIZE/2) and (stepStartingPos[0]>XSIZE/2):
            rewardTransition = GOAL_LINE_REWARD
        else:
            rewardTransition = 0.0

        # Calculate updated Q-Values
        costVector = list(qValues)
        if rewardTransition<0:
            costVector[targetDirDiscrete] = rewardTransition
            rw = rewardTransition
        else:
            costVector[targetDirDiscrete] = ALPHA_FACTOR*costVector[targetDirDiscrete] + (1.0-ALPHA_FACTOR)*rewardTransition
            rw = (1.0-ALPHA_FACTOR)*rewardTransition

        # Learn the updated Q-value.
        if isLearning:
            ds.clear()
            ds.addSample((stepStartingPos[0]/XSIZE, stepStartingPos[1]/YSIZE,math.sin(oldDir*0.25*math.pi),math.cos(oldDir*0.25*math.pi)),rw)
            trainer.setData(ds)
            trainer.trainEpochs( 1 )
            
    # ====================================
    # Final paint step
    # ====================================
    if pygame.display.get_active():        
        clock.tick(2)
        screen.blit(screenBuffer, (0, 0))
        pygame.display.flip()
    print "Trainer Alpha Value: ",trainer.descent.alpha
    
    # Let's look at the events. Key presses from 0 to 8 are possible, as well as space for switching between Q values and best direction painting. ESCape ends the program.
    for event in pygame.event.get():
                
        if event.type == pygame.locals.QUIT or (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_ESCAPE):           
            sys.exit(0)
        if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_SPACE):
            displayQValue = not displayQValue            
        if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_l):
            isLearning = not isLearning
        if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_RIGHT):
            displayDirection = (displayDirection + 1) % 8
        if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_LEFT):
            displayDirection = (displayDirection + 7) % 8

