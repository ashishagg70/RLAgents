# This is a sample Python script.
from enum import Enum
import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

class MDPInstance:
    def __init__(self):
        self.windShift=[0,0,0,1,1,1,2,2,1,0]
        self.start=(3,0)
        self.goal=(3,7)
        self.gridSize=(7,10)
    def nextState(self, state, action, stochastic):
        action=list(Actions)[action]
        #print("state: ", state)
        i, j = state // self.gridSize[1], state % self.gridSize[1]
        randNum = np.random.random()
        windEffect=self.windShift[j]
        if(self.windShift[j]>0 and stochastic==1):
            if(randNum<=0.33):
                windEffect=self.windShift[j]-1
            elif(randNum<=0.67):
                windEffect=self.windShift[j]+1
        #print(i, j, action, action.value)
        reward=-1
        nexti, nextj = i + action.value[0] - windEffect, j + action.value[1]
        nexti=np.maximum(0,nexti)
        nexti=np.minimum(self.gridSize[0]-1, nexti)

        nextj=np.maximum(0, nextj)
        nextj=np.minimum(self.gridSize[1]-1, nextj)

        return (self.gridSize[1] * nexti + nextj, reward)
class Actions(Enum):
    Up=(-1,0)
    Down=(1,0)
    Left=(0,-1)
    Right=(0,1)
    UpRight=(-1,1)
    UpLeft=(-1,-1)
    DownRight=(1,1)
    DownLeft=(1,-1)

