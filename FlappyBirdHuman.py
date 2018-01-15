# -------------------------
# Project: Human level control on Flappy Bird
# Author: whitezhang
# Date: 2018.1.1
# -------------------------

import cv2
import sys
import os
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQN_Nature import BrainDQN
import numpy as np
import pygame
from pygame.locals import *
import time
import datetime


# preprocess raw image to 80*80 gray image
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	return np.reshape(observation,(80,80,1))


def save2file(episodeMemory):
	episodeStr = []
	episodeLength = len(episodeMemory)
	for memory in episodeMemory:
		observation, action, reward, terminal = memory
		sObservation = ','.join(map(str, observation.flatten()))
		sAction = ','.join(map(str, action.flatten()))
		sReward = str(reward)
		sTerminal = '1' if terminal else '0'
		ptr = [sObservation, sAction, sReward, sTerminal]
		episodeStr.append('|'.join(ptr))

	if episodeLength < 30:
		return

	pathName = 'trainingData'
	if not os.path.exists(pathName):
		os.mkdir(pathName)
	ts = str(datetime.datetime.now().strftime('%Y%m%d-%H%M%S-' + str(episodeLength)))
	output = open(os.path.join(pathName, ts), 'w')
	output.writelines('\n'.join(episodeStr))


# action:
#	action[0] = 1 do nothing
#	action[1] = 1 flip
# terminal: True/False
def playFlappyBird():
	episodeMemory = []
	# Step 1: init BrainDQN
	actions = 2
	brain = BrainDQN(actions)
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
		action = np.array([1, 0])
		for event in pygame.event.get():
			if event.type == KEYDOWN and  event.key == K_UP:
				action = np.array([0, 1])
			if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
				pygame.quit()
				sys.exit()
		nextObservation, reward, terminal = flappyBird.frame_step(action)
		nextObservation = preprocess(nextObservation)
		if terminal:
			episodeMemory.append([nextObservation, action, reward, terminal])
			save2file(episodeMemory)
			episodeMemory = []
		else:
			episodeMemory.append([nextObservation, action, reward, terminal])

def main():
	playFlappyBird()

if __name__ == '__main__':
	main()
