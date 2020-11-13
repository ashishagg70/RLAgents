from Agents import Agents
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from customParser import getArg
class KingsMove(Enum):
    NoKingsMove="no kings move"
    KingsMove="kings move"
class Stochasticity(Enum):
    NoStochasticiy="no stochasticity of windeffect"
    Stochasticity="stochastic windeffect"
if __name__=='__main__':
    seedAvg=int(getArg('seed'))
    timeSteps=int(getArg('timeSteps'))
    epsilon = float(getArg('epsilon'))
    alpha = float(getArg('alpha'))
    kingsMove = int(getArg('kingsMove'))
    stochasticity = int(getArg('stochasticity'))
    agent = getArg('agent')
    fileName = getArg('fileName')
    if(kingsMove==1):
        numActions=8
    else:
        numActions=4
    print("running with alpha: {}, eplison: {}, with {} with {} taking average over {} seeds".format(alpha, epsilon, list(KingsMove)[kingsMove].value, list(Stochasticity)[stochasticity].value, seedAvg))
    if(agent==''):
        imgName="AllThreeWith{}AndWith{}.png".format(list(KingsMove)[kingsMove].name, list(Stochasticity)[stochasticity].name)
        print("generating graph and data for all three agents as no specific agent specified.")
    plt.title("graph generated with {} and {}".format(list(KingsMove)[kingsMove].value, list(Stochasticity)[stochasticity].value))
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")
    plt.figtext(.5, .8, "epsilon = {}\n alpha = {}".format(epsilon, alpha))
    if(agent=='' or agent.lower()=='sarsa'):
        if(agent!=''):
            imgName = "sarsaWith{}AndWith{}.png".format(list(KingsMove)[kingsMove].name,
                                                       list(Stochasticity)[stochasticity].name)
        print("running sarsa...")
        steps = np.zeros((seedAvg, timeSteps), dtype=int)
        for seed in range(seedAvg):
            agentObj = Agents(alpha, epsilon, numActions, timeSteps, seed, stochasticity)
            steps[seed]=agentObj.sarsa()
        meanEpisdoesOverSeedsSarsa=steps.mean(axis=0)
        plt.plot(range(timeSteps), meanEpisdoesOverSeedsSarsa, color="red", label='Sarsa')
    if (agent == '' or agent.lower() == 'qlearning'):
        if (agent != ''):
            imgName = "qLearningWith{}AndWith{}.png".format(list(KingsMove)[kingsMove].name,
                                                        list(Stochasticity)[stochasticity].name)
        print("running qlearning...")
        steps = np.zeros((seedAvg, timeSteps), dtype=int)
        for seed in range(seedAvg):
            agentObj = Agents(alpha, epsilon, numActions, timeSteps, seed, stochasticity)
            steps[seed] = agentObj.QLearning()
        meanEpisdoesOverSeedsQlearning = steps.mean(axis=0)
        plt.plot(range(timeSteps), meanEpisdoesOverSeedsQlearning, color="blue", label="QLearning")
    if(agent == '' or agent.lower() == 'expectedsarsa'):
        if (agent != ''):
            imgName = "expectedSarsaWith{}AndWith{}.png".format(list(KingsMove)[kingsMove].name,
                                                        list(Stochasticity)[stochasticity].name)
        print("running expectedSarsa...")
        steps = np.zeros((seedAvg, timeSteps), dtype=int)
        for seed in range(seedAvg):
            agentObj = Agents(alpha, epsilon, numActions, timeSteps, seed, stochasticity)
            steps[seed] = agentObj.expectedSarsa()
        meanEpisdoesOverSeedsExpectedSarsa = steps.mean(axis=0)
        plt.plot(range(timeSteps), meanEpisdoesOverSeedsExpectedSarsa, color="green", label="Expected Sarsa")
    plt.legend(loc='upper left')
    if(fileName==''):
        plt.savefig(imgName)
        print('plot saved to file ', imgName)
    else:
        plt.savefig(fileName)
        print('plot saved to file ', fileName)
    #plt.show()
    #print(meanStepsOverSeeds)
