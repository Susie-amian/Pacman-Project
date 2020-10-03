# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'QlearningAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class QlearningAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    self.weights = util.Counter()
    self.discount = 0.8
    self.alpha = 1.0
    self.epsilon = 0.05
    


    '''
    Your initialization code goes here, if you need any.
    '''

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    foodLeft = len(self.getFood(gameState).asList())

    # Case 1: food left <= 2
    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          ToAct = action
          bestDist = dist

    # Case 2: Otherwise
    else:
      bestQvals, bestActions = self.getQvalActions(self, gameState, actions)
      # Eploration-exploitation
      ToAct = epsilonGready(epsilon, bestActions)
      
    return ToAct

  def epsilonGready(self, e, exploreActions, exploitActions):
    """
    Returns an action using epsilon greedy method
    """
    exploit = util.flipCoin(e)
    if exploit and exploitActions:
      ToAct = random.choice(exploitActions)
    else:
      ToAct = random.choice(exploreActions)
    return ToAct


  def getBestQvalActions(self, gameState):
    """
    Returns the maximum Q values and their corresponding actions
    """
    #list of Qvals
    Qvals = [self.getQvals(gameState, act) for act in actions]

    maxQvals = max(Qvals)
    bestActions = [act for act, val in zip(actions, Qvals) if val == maxQvals]
    
    return maxQvals, random.choice(bestActions)

  def getQvals(self, gameState, action):
    """
    Returns the calculated Q values
    Version 1: Use a linear function in features and weights
    Qval = dotProduct(features, weights)
    """
    features = self.getFeatures(gameState, action) # return features counter
    weights = self.getWeights() # weights counter
    
    # dot product of features and weights
    Qval = features*weights
    return Qval

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    
    # score
    features['successorScore'] = self.getScore(successor)

    # distance to capsule

    # distance to food

    # total steps left

    #


    return features

  def getWeights(self):
    return self.weights

  def getReward(self, gameState, toAct):
    # init reward
    reward = 0

    cur_pos = gameState.getAgentPosition(self.index)
    suc_state = gameState.generateSuccessor(self.index, toAct)
    suc_pos = suc_state.getAgentState(self.index).getPosition()

    # better score
    if gameState.getScore() < suc_state.getSccore():
      reward += 20

    # closer distance to capsule

    # closer distance to food


  def updateWeights(self,gameState,feature, action,nextState,reward):
    
    oldQ = self.getQvals(gameState, action)
    newMaxQval, _ = self.getBestQvalActions(nextState)
    learnedWeight = self.alpha*(reward + self.discount * newMaxQval - oldQ)
    features = self.getFeatures(gameState, action)  #counter
    self.weights += learnedWeight * features
     
  
  def initWeights():
    pass
  
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor