import time
import pygame, sys
import numpy as np
import matplotlib.pyplot as plt

class Snake:
    def __init__(self, x = 0, y = 0, blockSize = 20):
        self.position  = [[x, y]]
        self.tail      = [[x, y]]
        self.blockSize = blockSize
        self.direction = 0

    def move(self):
        self.tail = self.position[-1].copy()
        for ind in np.arange(len(self.position) - 1, 0, -1):
            self.position[ind] = self.position[ind - 1].copy()

        if (self.direction == 0):
            self.position[0][0] += self.blockSize
        if (self.direction == 1):
            self.position[0][1] += self.blockSize
        if (self.direction == 2):
            self.position[0][0] -= self.blockSize
        if (self.direction == 3):
            self.position[0][1] -= self.blockSize
            
class Game:

    def __init__(self, width = 1280, height = 720, speed = 1, snakeColor = (0, 255, 0), foodColor = (255, 0, 0)):
        self.score        = 0
        self.n_actions    = 4
        self.gameOver     = False
        self.player       = Snake()
        height = self.player.blockSize * int(height / self.player.blockSize)
        width  = self.player.blockSize * int(width  / self.player.blockSize)
        
        self.screenHeight = height
        self.screenWidth  = width
        self.speed        = speed
        self.screenSize   = width, height
        self.snakeColor   = snakeColor
        self.foodColor    = foodColor
        self.clock        = pygame.time.Clock()
        # self.screen       = pygame.display.set_mode(self.screenSize)
        self.newFood()
        
    def newFood(self):
        blockSize = self.player.blockSize
        height    = self.screenHeight
        width     = self.screenWidth
        self.foodPosition = [blockSize * np.random.randint(0, width / blockSize), blockSize * np.random.randint(0, height / blockSize)]
        
    def checkPosition(self):
        headPosition = self.player.position[0]
        height       = self.screenHeight
        width        = self.screenWidth
        if headPosition[0] < 0 or headPosition[0] > width:
            self.gameOver = True
        if headPosition[1] < 0 or headPosition[1] > height:
            self.gameOver = True
        if len(np.unique(self.player.position, axis = 0)) != len(self.player.position):
            self.gameOver = True
        if headPosition == self.foodPosition:
            self.score += 1
            self.player.position.append(self.player.tail.copy())
            self.newFood()
            return True
        return False

    # def render(self):
    #     self.screen.fill((255, 255, 255))
    #     # for i in self.player.position:
    #         # pygame.draw.rect(self.screen, self.snakeColor, [i[0], i[1], self.player.blockSize, self.player.blockSize])
    #     # pygame.draw.rect(self.screen, self.foodColor, [self.foodPosition[0], self.foodPosition[1], self.player.blockSize, self.player.blockSize])
    #     # pygame.display.update()
    #     time.sleep(1)

    def step(self, action):
        state  = self.getState()
        reward = self.getReward()
        done   = self.gameOver

        self.player.move()
        # print(self.player.position[0])
        return state, reward, done

    def getState(self):
        x    = self.foodPosition
        y    = self.player.position[0]
        z    = self.player.position
        size = self.player.blockSize
        return [x[0] <= y[0], x[1] >= y[1], x[0] == y[0], x[1] == y[1], self.player.direction, int([y[0] - size, y[1]] in z), int([y[0] + size, y[1]] in z), int([y[0], y[1] - size] in z), int([y[0], y[1] + size] in z)]

    def getReward(self):
        reward = -1
        if (self.checkPosition()):
            reward = 0.5
        return reward

    def reset(self):
        self.__init__(self.screenWidth, self.screenHeight, self.speed, self.snakeColor, self.foodColor)
        return self.getState()

class CEM:
    def __init__(self):
        pass