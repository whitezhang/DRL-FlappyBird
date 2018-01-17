# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: whitezhang
# Date: 2018.1.1
# -------------------------

import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQN_Nature import BrainDQN
import numpy as np
import os

# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(80,80,1))


def preTraining(brain, pathname='./trainingData/'):
    for root, dirs, files in os.walk(pathname):
        for name in files:
            filename = os.path.join(root, name)
            with open(filename) as fin:
                for line in fin:
                    sObservation, sAction, sReward, sTerminal = line.split('|')
                    observation = np.reshape(list(map(int, sObservation.split(','))), (80, 80, 1))
                    actionTmp = list(map(int, sAction.split(',')))
                    action = np.array([actionTmp[0], actionTmp[1]])
                    reward = float(sReward)
                    terminal = True if sTerminal == 1 else False
                    brain.setPerception(observation, action, reward, terminal)
    return brain


def playFlappyBird():
    # Step 1: init BrainDQN
    actions = 2
    brain = BrainDQN(actions)
    brain = preTraining(brain)
    # Step 2: init Flappy Bird Game
    flappyBird = game.GameState()
    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([1,0])  # do nothing
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
    brain.setInitState(observation0)

    # Step 3.2: run the game
    while 1!= 0:
        action = brain.getAction()
        nextObservation, reward, terminal = flappyBird.frame_step(action)
        nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation,action,reward,terminal)


def main():
    playFlappyBird()


if __name__ == '__main__':
    main()
