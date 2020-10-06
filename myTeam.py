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
import distanceCalculator
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'QlearningAgent', second = 'QlearningAgent'):
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
    self.start = gameState.getAgentPosition(self.index)

    self.weights = util.Counter()
    self.discount = 0.8
    self.alpha = 1.0
    self.epsilon = 0.9

    self.episodeRewards = 0.0
    self.lastAction = None
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)
    self.distancer.getMazeDistance()
    #self.totalFood = len(self.getFood(self.start).asList())

    self.goalFood, self.DistGoalFood = self.getDistToFood(gameState)


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


    '''Update weight based on previous completed action and state'''
    if self.lastAction != None:
      lastState = self.getPreviousObservation()
      features = self.getFeatures(lastState, self.lastAction)
      
      self.updateWeights(lastState,features, self.lastAction, gameState)
   

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

      _, bestAction = self.getBestQvalAction(gameState)
      # Eploration-exploitation
      ToAct = self.epsilonGreedy(self.epsilon, bestAction, actions)

    self.lastAction = ToAct
    print('toact',ToAct)
    return ToAct

  def epsilonGreedy(self, e, exploitAction, exploreActions):
    """
    Returns an action using epsilon greedy method
    """
    exploit = util.flipCoin(e)
    if exploit and exploitAction:
      ToAct = exploitAction
    else:
      ToAct = random.choice(exploreActions)
    return ToAct


  def getBestQvalAction(self, gameState):
    """
    Returns the maximum Q values and their corresponding actions
    """
    actions = gameState.getLegalActions(self.index)
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

    # distance to closest food
    _, features['distToFood'] = self.getDistToFood(successor)
    

    # total steps left

    #


    return features

  def getDistToFood(self, currentState):
    pos = currentState.getAgentState(self.index).getPosition()
    foodList =  self.getFood(currentState).asList()
    min_dist = 9999
    #print('209',foodList)
    for food in foodList:

      
      dist = self.getMazeDistance(food, pos)
      if dist < min_dist:
        min_dist = dist
        food_pos = food
    return food_pos, min_dist

  def getWeights(self):
    return self.weights

  def getReward(self, gameState, toAct):
    # init reward
    reward = 0
    curState = gameState
    curPos = gameState.getAgentPosition(self.index)
    curFoodNum = len(self.getFood(curState).asList())

    sucState = gameState.generateSuccessor(self.index, toAct)
    sucPos = sucState.getAgentState(self.index).getPosition()

    prevState = self.getPreviousObservation()
    prevPos = prevState.getAgentPosition(self.index)
    prevFoodNum = len(self.getFood(prevState).asList())

    
    
    
    # better score
    if gameState.getScore() < sucState.getScore():
      reward += 20
    
    # food difference
    foodDifference = curFoodNum - prevFoodNum
    if (foodDifference > 0):
      reward += 5
    elif (foodDifference < 0):
      reward += foodDifference*2

    # closer distance to capsule

    # closer distance to food
    if self.goalFood == curPos:
      self.goalFood, curDist = self.getDistToFood(curState)
    else:
      curDist = self.getMazeDistance(curPos, self.goalFood)
   
    reward += self.getFoodProximityReward(prevPos, curDist, self.goalFood)
    print('THIS IS OUR PROXIMITY REWARD', self.getFoodProximityReward(prevPos, curDist, self.goalFood))
    print('OUR TOTAL REWARD', reward)
    return reward

  def getFoodProximityReward(self, prevPos, curDist, goalFood):
    #print('======', prevPos, self.goalFood)
    prevDist = self.getMazeDistance(prevPos, goalFood)
    return 100*(prevDist-curDist)    

  def updateWeights(self, gameState,feature, action,nextState):
    reward = self.getReward(gameState, action) #value
    oldQ = self.getQvals(gameState, action) #value
    newMaxQval, _ = self.getBestQvalAction(nextState) #value
    learnedWeight = self.alpha*(reward + self.discount * newMaxQval - oldQ)  #value 
    features = self.getFeatures(gameState, action)  #counter
    for key in features.keys():
      self.weights[key] += learnedWeight * features[key]
     
  
  def initWeights(self):
    pass
  
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  