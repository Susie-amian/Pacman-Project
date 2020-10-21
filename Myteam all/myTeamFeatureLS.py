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

from baselineTeam import DefensiveReflexAgent
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import game

arguments = {}

CHEAT = False
beliefs = []
beliefsInitialized = []
MINIMUM_PROBABILITY = .0001
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'QlearningAgent', second = 'DefensiveReflexAgent',**args):
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
  print("AA")
  if 'numTraining' in args:
    print('a')
    arguments['numTraining'] = args['numTraining']
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
  def __init__(self, index):
    CaptureAgent.__init__(self, index)
    self.epsilon = 0.9
    self.alpha = 0.2
    self.discount = 0.7

    self.episodesSoFar = 0
    self.numTraining = 0

    self.weights = util.Counter()

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

    '''
    Your initialization code goes here, if you need any.
    '''

    

    self.start = gameState.getAgentPosition(self.index)
    
    
    self.totalFoodList = self.getFood(gameState).asList()
    self.eatenFood = []

    # recording dict
    self.weights = util.Counter()
    

    #self.weights['successorScore'] = 1
    #self.weights['distanceToFood'] = -1
    self.weights['distToFood'] = 1
    self.weights['stop'] = -2
    #self.weights['eatenFood'] = 10
    """self.weights = util.Counter({
      'successorScore': 100,
      'distToFood': 10,
      'ghostDistance': 10,
      'stop': -1,
      'powerPelletValue': 10,
      'backToSafeZone': 1,
      'numInvaders': -10,
      'onDefense': 0.5,
      'invaderDistance': -10,
      'stop': -10, 
      'reverse': -2,
      'eatenFood':10
    })"""

    #self.goalFood, self.DistGoalFood = self.getDistToFood(gameState)
    
    # 
    self.threatenedDistance = 5
    self.minPelletsToCashIn = 5
    self.distanceToTrackPowerPelletValue = 3
    # dictionary of (position) -> [action, ...]
    # populated as we go along; to use this, call self.getLegalActions(gameState)
    self.legalActionMap = {}
    self.legalPositionsInitialized = False
    self.lastNumReturnedPellets = 0.0

    self.defenseTimer = 0.0
    self.initFoodList = self.getFood(gameState).asList()
    #self.eatenFood = set()
    self.lastAction = None
    self.lastState = None




    # get layout frontier
    self.width = gameState.data.layout.width
    self.height = gameState.data.layout.height
    #print([(i, j) for i in range(self.width) for j in range(self.height)])

    self.midPointLeft = int((self.width / 2.0)-1)
    self.midPointRight = int((self.width / 2.0)+1)
    self.isRed = gameState.isOnRedTeam(self.index)

    if self.isRed:
      self.midPoint = int(self.midPointLeft)
      self.enemyCells = []
      for i in range(self.midPointRight, self.width):
        for j in range(self.height):
          if not gameState.hasWall(i, j):
            
            self.enemyCells.append((i, j))
      
    else:
      self.midPoint = int(self.midPointRight)
      self.enemyCells = []
      for i in range(self.midPointLeft):
        for j in range(self.height):

          if not gameState.hasWall(i, j):
            
            self.enemyCells.append((int(i), int(j)))

    self.frontierPoints = [(self.midPoint, int(i)) for i in range(self.height) if not gameState.hasWall(self.midPoint, i)]
    self.distToHome = self.getDistToHome(self.frontierPoints, self.enemyCells)

  def chooseAction(self, gameState):
    if self.lastAction != None:
      #lastState = self.getPreviousObservation()
      
      #features = self.getFeatures(self.lastState, self.lastAction)
      
      self.update(self.lastState,self.lastAction, gameState)
      print('118',gameState.getAgentState(self.index).numCarrying)
  
      print('139',self.lastState, gameState)
      print('140',self.lastState == self.getPreviousObservation(), gameState == self.getCurrentObservation())

    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''
    foodLeft = len(self.getFood(gameState).asList())

    values = [self.getQvals(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = self.getBestActions(actions, maxValue, values)

    # number of food eaten before go home
    #self.eatenFood = self.currentEatenFood(gameState)
    
     
    # if go home, eaten food num start from 0 
    """if not myState.isPacman:
      #features['eatenFoodNum'] = 0
      self.eatenFood = set()"""
    
    # Case 1: food left <= 2
    if foodLeft <= 2:
        bestDist = 9999
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(self.start, pos2)
            if dist < bestDist:
                actionToReturn = action
                bestDist = dist
    else:
        actionToReturn = self.epsilonGreedy(bestActions, actions)

    # get reward
    successor = gameState.generateSuccessor(self.index, actionToReturn)

   
    self.lastAction = actionToReturn

    self.lastState = gameState
    
    return actionToReturn

  def epsilonGreedy(self, exploitAction, exploreActions):
    """
    Returns an action using epsilon greedy method
    """
    exploit = util.flipCoin(self.epsilon)
    if exploit and exploitAction:
      ToAct = random.choice(exploitAction)

    else:
      ToAct = random.choice(exploreActions)

    return ToAct

  def getBestActions(self, actions, maxval, vals):
    return [act for act, val in zip(actions, vals) if val == maxval]

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    currentState = gameState.getAgentPosition(self.index)
    successor = gameState.generateSuccessor(self.index, action)
    pos2 = successor.getAgentState(self.index).getPosition()
    x,y = pos2
    pos2 = (int(x),int(y))

    if pos2 != util.nearestPoint(pos2):
        # Only half a grid position was covered
        return successor.generateSuccessor(self.index, action)
    else:
        return successor

  def getQvals(self, gameState, action):
    """
    Returns the calculated Q values
    Version 1: Use a linear function in features and weights
    Qval = dotProduct(features, weights)
    """
    features = self.getFeatures(gameState, action) # return features counter
    weights = self.getWeights() # weights counter

    Qval = features*weights

    return Qval

 
  def getFeatures(self, gameState, action):
    features = util.Counter()
    curState = gameState.getAgentState(self.index)
    curPos = gameState.getAgentPosition(self.index)
    curFoodList = self.getFood(gameState).asList()

    successor = self.getSuccessor(gameState, action)
    sucState = successor.getAgentState(self.index)
    sucFoodList = self.getFood(successor).asList()
    sucPos = successor.getAgentPosition(self.index)
    # Score
    #features['successorScore'] = self.getScore(successor)  # self.getScore(successor)
    #features['successorScore'] = len(self.initFoodList)-len(sucFoodList)
      
    # Distance to closest food
    _, minDist = self.getDistToFood(successor,sucFoodList)
    features['distToFood'] = 1/minDist

    curCarry = curState.numCarrying
    sucCarry = sucState.numCarrying
    #diffFoodList = list(set(curFoodList) - set(sucFoodList))

    features['successorScore'] = curCarry
    
    if sucCarry - curCarry:
      print('ADDING CARRYING 319', sucCarry - curCarry)
      features['distToFood'] = 1.5
      #features['eatenFood'] = 1
    



    # need to cash in
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",self.getCashInValue(curPos, curState))
    features['cashIn'] = 0 
    if self.getCashInValue(curPos, curState) != 0 :
      features['cashIn'] = self.getCashInValue(curPos, curState)[0]/5
 

    # Get all enemies
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    nonScaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
    scaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]

    # Computes whether we're on defense (1) or offense (0)
    """features['onDefense'] = 1
    if myState.isPacman: 
      features['onDefense'] = 0
    """
    # Computes distance to invaders we can see
    #features['numInvaders'] = len(invaders)
    numInvaders = len(invaders)
    
    """if len(invaders) > 0:
      dists = [self.getMazeDistance(sucPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)"""

    # Heavily prioritize not stopping
    if action == Directions.STOP: 
      features['stop'] = 1
    # reverse
    

    # distance to powerPellet
    #features['powerPelletValue'] = self.getPowerPelletValue(sucPos, successor, scaredGhosts)

    
    # Adding value for going back home
    features['backToSafeZone'] += self.getBackToStartDistance(sucPos, features['ghostDistance'])

     


    


    



    #distance to the ghost
    #features['ghostDistance']

    if self.shouldRunHome(gameState):
      features['backToSafeZone'] = self.getMazeDistance(self.start, sucPos) 

    return features

  def getCashInValue(self, myPos, state):
      # if we have enough pellets, attempt to cash in
      if state.numCarrying >= self.minPelletsToCashIn:
        return self.distToHome[myPos]
      else:
        return 0

  def getDistToHome(self, home, possibleLocs):
    distToHome = util.Counter()
    for loc in possibleLocs:
      minDist = 9999
      for h in home:
        curDist = (self.getMazeDistance(h, loc))
        if curDist < minDist:
          minDist = curDist
          distToHome[loc] = (curDist, h)
    return distToHome
    
  # If there are not any scared ghosts, then we value eating pellets
  def getPowerPelletValue(self, myPos, successor, scaredGhosts):
      powerPellets = self.getCapsules(successor)
      minDistance = 0
      if len(powerPellets) > 0 and len(scaredGhosts) == 0:
        distances = [self.getMazeDistance(myPos, pellet) for pellet in powerPellets]
        minDistance = min(distances)
      return max(self.distanceToTrackPowerPelletValue - minDistance, 0)


  # If we are near the end of the game, we should go home
  def shouldRunHome(self, gameState):
    winningBy = self.getWinningBy(gameState)
    numCarrying = gameState.getAgentState(self.index).numCarrying
    PANIC_TIME = 80
    return (gameState.data.timeleft < PANIC_TIME 
      and winningBy <= 0 
      and numCarrying > 0 
      and numCarrying >= abs(winningBy))


  def getBackToStartDistance(self, myPos, smallestGhostPosition):
    if smallestGhostPosition > self.threatenedDistance or smallestGhostPosition == 0:
      return 0
    else:
      return self.getMazeDistance(self.start, myPos) 

  def getLegalActions(self, gameState):
    """
    legal action getter that favors 
    returns list of legal actions for Pacman in the given state
    """
    currentPos = gameState.getAgentState(self.index).getPosition()
    if currentPos not in self.legalActionMap:
      self.legalActionMap[currentPos] = gameState.getLegalActions(self.index)
    return self.legalActionMap[currentPos]

  def getWinningBy(self, gameState):
    if self.red:
      return gameState.getScore()
    else:
      return -1 * gameState.getScore()

  
  

  def getDistToFood(self, currentState, foodList):
    pos = currentState.getAgentState(self.index).getPosition()
    #foodList =  self.getFood(currentState).asList()
    min_dist = 9999

    for food in foodList:
   
      dist = self.getMazeDistance(food, pos)
      if dist < min_dist:
        min_dist = dist
        food_pos = food

    return food_pos, min_dist

  def getWeights(self):
    return self.weights

  def computeValueFromQValues(self, gameState):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"

    actions = gameState.getLegalActions(self.index)
    values = [self.getQvals(gameState, a) for a in actions]
    maxQValue = max(values)

    return maxQValue

  def getReward(self, gameState, toAct, nextState):
    # init reward
    reward = 0
    curState = gameState
    curPos = curState.getAgentPosition(self.index)
    curFoodNum = len(self.getFood(curState).asList())

    sucState = nextState
    sucPos = sucState.getAgentState(self.index).getPosition()
    sucFoodNum = len(self.getFood(sucState).asList())

    # better score
    if sucState.getScore() > curState.getScore():
      reward += 5

    #does the agent need cash in   
    needCashIn = False
    curCarryingFood = gameState.getAgentState(self.index).numCarrying
    if curCarryingFood >= self.minPelletsToCashIn:
      needCashIn = True

    # food difference
    totalFoodDiff = len(self.initFoodList) - sucFoodNum
    foodDifference = curFoodNum - sucFoodNum
    if not needCashIn:
      if (foodDifference > 0):
        reward += foodDifference
      elif (foodDifference < 0):
        reward += foodDifference*2

    # closer distance to capsule

    # closer distance to food
    foodList = self.getFood(gameState).asList()
    curToFood = abs(min([self.getMazeDistance(curPos, food) for food in foodList]))
    sucToFood = abs(min([self.getMazeDistance(sucPos, food) for food in foodList]))
    if not needCashIn:
      if curToFood > sucToFood:
        reward += 1
      else:
          reward -= 1
    """if self.goalFood == sucPos:
      self.goalFood, sucDist = self.getDistToFood(sucState)
    else:
      sucDist = self.getMazeDistance(sucPos, self.goalFood)
   
    reward += self.getFoodProximityReward(curPos, sucDist, self.goalFood)"""

    if needCashIn:
      curDistToHome = self.distToHome[curPos]
      sucDistToHome = self.distToHome[sucPos]
      if sucDistToHome[0] < curDistToHome[0]:
        reward += 10
      else:
        reward -= 1
      
    return reward



  def getFoodProximityReward(self, prevPos, curDist, goalFood):

    prevDist = self.getMazeDistance(prevPos, goalFood)
    return 100*(prevDist-curDist) 

  def update(self, gameState, action, nextState):
    
    reward = self.getReward(gameState, action, nextState) #value
    
    oldQ = self.getQvals(gameState, action) #value
    newMaxQval= self.computeValueFromQValues(nextState) #value
    learnedWeight = self.alpha*(reward + self.discount * newMaxQval - oldQ)  #value 
    features = self.getFeatures(gameState, action)  #counter

    pos = gameState.getAgentState(self.index).getPosition()
   
    

    for key in features.keys():
      self.weights[key] += learnedWeight * features[key]
    print("===FEATURE===388",pos,self.index)
    print(self.weights)
    print(features)
    print(oldQ, newMaxQval)
    print(reward)
  
  def isTraining(self):
    return self.episodesSoFar < self.numTraining

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    CaptureAgent.final(self, state)
    #if self.isTraining() and DEBUG:
      #print "END WEIGHTS"
      #print self.weights
    self.episodesSoFar += 1