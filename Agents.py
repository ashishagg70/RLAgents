from Instance import Actions
from Instance import MDPInstance
import numpy as np
import matplotlib.pyplot as plt
class Agents:
    def __init__(self,alpha, epsilon, numActions, numSteps, seed, stochasticity):
        np.random.seed(seed)
        self.instance=MDPInstance()
        gridSize=self.instance.gridSize
        start, goal=self.instance.start, self.instance.goal
        self.alpha=alpha
        self.gamma=1
        self.epsilon=epsilon
        self.numActions=numActions
        self.numStates=gridSize[0]*gridSize[1]
        self.goal=gridSize[1]*goal[0]+goal[1]
        self.start=gridSize[1]*start[0]+start[1]
        self.T=numSteps
        self.stochasticity=stochasticity
        self.Q_s_a=np.zeros((self.numStates, self.numActions))
    def sarsa(self):
        episodes=[0]*self.T
        episodeCount=0
        i=0
        while i <=self.T:
            s=self.start
            action=self.chooseAction(s)
            while(s!=self.goal):
                nextState, reward=self.instance.nextState(s,action, self.stochasticity)
                nextAction=self.chooseAction(nextState)
                self.Q_s_a[s][action]=self.Q_s_a[s][action]+self.alpha*(reward+self.gamma*self.Q_s_a[nextState][nextAction]-self.Q_s_a[s][action])
                #print(self.Q_s_a)
                s, action=nextState, nextAction
                if(i<self.T):
                    episodes[i]=episodeCount
                i+=1
            episodeCount+=1
            #print("out")
        return episodes
    def QLearning(self):
        episodes = [0] * self.T
        episodeCount = 0
        i = 0
        while i <= self.T:
            s = self.start
            while (s != self.goal):
                action = self.chooseAction(s)
                nextState, reward = self.instance.nextState(s, action, self.stochasticity)
                self.Q_s_a[s][action] = self.Q_s_a[s][action] + self.alpha * (
                            reward + self.gamma * np.max(self.Q_s_a[nextState]) - self.Q_s_a[s][action])
                # print(self.Q_s_a)
                s = nextState
                if (i < self.T):
                    episodes[i] = episodeCount
                i += 1
            episodeCount += 1
        return episodes

    def expectedSarsa(self):
        episodes = [0] * self.T
        episodeCount = 0
        i = 0
        while i <= self.T:
            s = self.start
            while (s != self.goal):
                action = self.chooseAction(s)
                nextState, reward = self.instance.nextState(s, action, self.stochasticity)
                self.Q_s_a[s][action] = self.Q_s_a[s][action] + self.alpha * (
                            reward + self.gamma * self.calculateExpectedActionValue(nextState) - self.Q_s_a[s][action])
                s = nextState
                if (i < self.T):
                    episodes[i] = episodeCount
                i += 1
            episodeCount += 1
        return episodes

    def calculateExpectedActionValue(self, nextState):
        policy=np.zeros(self.numActions)
        policy.fill(self.epsilon/self.numActions)
        greedyAction=np.random.choice(np.where(self.Q_s_a[nextState]==np.max(self.Q_s_a[nextState]))[0])
        policy[greedyAction] += 1 - self.epsilon
        #greedyActions=np.where(self.Q_s_a[nextState]==np.max(self.Q_s_a[nextState]))[0]
        #policy[greedyActions]+=(1-self.epsilon)/len(greedyActions)
        return np.sum(policy*self.Q_s_a[nextState])


    def chooseAction(self, s):
        randE = np.random.random()
        if (randE <= self.epsilon):  # explore
            action = np.random.randint(self.numActions)
        else:
            action = np.argmax(self.Q_s_a[s])
        return action












