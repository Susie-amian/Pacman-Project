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
from capture import GameState
import distanceCalculator
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ClassicPlanAgent', second = 'DefensiveReflexAgent'):
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
class ClassicPlanAgent(CaptureAgent):
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

    '''
    Your initialization code goes here, if you need any.
    '''
    self.enemies = self.getOpponents(gameState)
    self.start = gameState.getAgentPosition(self.index)
    
    self.totalFoodNum = len(self.getFood(gameState).asList())
    self.minPelletsToCashIn = int(self.totalFoodNum/4)
    # Game information
    self.isRed = gameState.isOnRedTeam(self.index)
    self.myFriend = gameState.getRedTeamIndices() if gameState.isOnRedTeam(self.index) else gameState.getBlueTeamIndices()
    self.myFriend.remove(self.index)
    self.myFriend = self.myFriend[0]

    # Board information
    self.width = gameState.data.layout.width
    self.height = gameState.data.layout.height
    self.midWidth = self.width/2
    self.midHeight = self.height/2

    self.midPointLeft = int((self.width / 2.0)-1)
    self.midPointRight = int((self.width / 2.0)+1)

    self.midPoint, self.midPointEnemy, self.enemyCells = self.getBoardInfo(gameState)

    self.frontierPoints = [(self.midPoint, int(i)) for i in range(self.height) if not gameState.hasWall(self.midPoint, i)]
    self.frontierPointsEnemy = [(self.midPointEnemy, int(i)) for i in range(self.height) if not gameState.hasWall(self.midPointEnemy, i)]

    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.myCells = [c for c in self.legalPositions if c not in self.enemyCells]
    
    # A dictionary for the closest distance from any enemy cell to home
    self.distToHome = self.getDistToHome(self.frontierPoints, self.enemyCells)
    # A sequence of action to the closest enemy cell
    self.frontierState, self.actionFrontier = self.toFrontier(gameState, self.frontierPoints, self.start)
    # get shortest path from frontier to power capsule
    if self.getCapsules(gameState):
      minDist = 9999
      for cap in self.getCapsules(gameState):
        if minDist > self.getMazeDistance(cap, self.start):
          minDist = self.getMazeDistance(cap, self.start)
          self.capsulePosition = cap

    self.closestFrontier = self.frontierState.getAgentPosition(self.index)
    if self.closestFrontier in self.frontierPoints[:int(len(self.frontierPoints)/2)]:
      self.favoredY = 0.0
    else:
      self.favoredY = self.height
    self.capsulState, self.actionCapsule = self.getBfsPath(self.frontierState, self.capsulePosition, self.closestFrontier)
    # minimax initial set up
    self.miniMaxDepth = 4
    # Initialisation for hyper-parameters
    self.epsilon = 0.75

    # enemies
    self.enemies = self.getOpponents(gameState)
    self.belief = {}    # our belief of where the ghost might be
    for enemy in self.enemies:
      self.belief[enemy] = util.Counter()
    #print(self.belief[self.enemies[0]])

  def getBoardInfo(self, gameState):
    """
    This function provides information of the mid point of both sides and the enemy grid
    """
    if self.isRed:
      midPoint = int(self.midPointLeft)
      midPointEnemy = int(self.midPointRight)
      enemyCells = []
      for i in range(self.midPointRight-1, self.width):
        for j in range(self.height):
          if not gameState.hasWall(i, j):
            enemyCells.append((i, j))
      
    else:
      midPoint = int(self.midPointRight)
      midPointEnemy = int(self.midPointLeft)
      enemyCells = []
      for i in range(self.midPointLeft):
        for j in range(self.height):
          if not gameState.hasWall(i, j):
            enemyCells.append((int(i), int(j)))
    return midPoint, midPointEnemy, enemyCells

  def getDistToHome(self, home, possibleLocs):
    distToHome = util.Counter()
    for loc in possibleLocs:
      mindist = 9999
      for h in home:
        curDist = (self.getMazeDistance(h, loc))
        if curDist < mindist:
          mindist = curDist
          distToHome[loc] = (curDist, h)
    return distToHome

  def toFrontier(self, gameState, frontierPoints, start):
    """
    Returns the closest path to home frontier
    """
    minDist = 9999
    for point in frontierPoints:
      currentDist = self.getMazeDistance(point, start)
      if currentDist < minDist:
        minDist = currentDist
        minPosition = point
    actionSeq = self.getBfsPath(gameState, minPosition, start)
    return actionSeq

  def getBfsPath(self, gameState, end, start):
    """
    Helper function for toFrontier
    Using BFS to search path
    """
    explored = [start]
    states = util.Queue()
    stateRecord = (gameState, [])
    states.push(stateRecord)
    cur_pos = start
    while not states.isEmpty():
      state, action = states.pop()
      cur_pos = state.getAgentPosition(self.index)

      if cur_pos == end:
        return state, action
      
      legalActions = state.getLegalActions(self.index)

      for a in legalActions:
        successor = state.generateSuccessor(self.index, a)
        coor = successor.getAgentPosition(self.index)
        if coor not in explored:
          explored.append(coor)
          states.push((successor, action+[a]))
   
  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''
    # INIT VARIABLES TO USE
    gstPos = self.checkStateSafe(gameState)
    myPos = gameState.getAgentPosition(self.index)
    Pacman = gameState.getAgentState(self.index).isPacman
    myState = gameState.getAgentState(self.index)
    defendFood = self.getFoodYouAreDefending(gameState).asList()
    timeLeft = gameState.data.timeleft    # for debug
    
    # === ABOUT-TO-LOSE SENARIO ===
    if len(defendFood) <= 4:
      
      values = [self.evaluatePatrol(gameState, a) for a in actions]
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      toAct = random.choice(bestActions)
      #print('=== 228 ===', self.index, toAct, timeLeft)
      return toAct

    # === NORMAL SENARIO ===
    # ACTION STEP 1: reach frontier first
    if self.actionFrontier:
      toAct = self.actionFrontier.pop(0)
      #print('=== 235 ===', self.index, toAct, timeLeft)
      return toAct

    # ACTION STEP 2: after reaching frontier
    if self.actionCapsule:
      toAct = self.actionCapsule.pop(0)

      # check if planned step is safe
      gstPos = self.checkStateSafe(gameState)
      if not gstPos:    # safe, ghost not detected
        #print('=== 245 ===', self.index, toAct, timeLeft)
        return toAct

      # otherwise, ghost detected, run.
      self.actionCapsule = []    # ditch all pre-planed path

    # === REACHED-AN-IMPASSE SENARIO ===
    isInImpasse = self.reachedImpasse(gameState, myPos)
    if isInImpasse:
      values = [self.evaluateImpasse(gameState, a) for a in actions]
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      bestAction = random.choice(bestActions)
      #print('=== 259 ===', self.index, bestAction, timeLeft)
      return bestAction
    
    # === DETECTS GHOST === USE MINIMAX ===
    if gstPos and Pacman:
      enermyIndex = [tup[0] for tup in gstPos]
      if len(enermyIndex) == 1:
        allIndexes = [self.index, enermyIndex[0]]
        depth = self.miniMaxDepth
        v, toAct = self.max2(gameState, depth, self.index, allIndexes)
        # new minimax 2 
      else:
        #toAct = self.getMiniMaxAction(gameState, myPos, enemyPos)
        toAct = random.choice(actions)
      #print('=== 274 ===', self.index, toAct, timeLeft)
      return toAct

    # ACTION SENARIO 4: if food more than threshold, need to cash in
    goHome = self.needToCashIn(myPos, myState, self.minPelletsToCashIn)
    isHome = self.distToHome[myPos]
    isCloseToFood = self.isCloseToFood(gameState, actions)
    if goHome and (isHome) and (not isCloseToFood):
      dist, end = self.distToHome[myPos]
      _, path = self.getBfsPath(gameState, end, myPos)
      toAct = path.pop(0)
      #print('=== 285 ===', self.index, toAct, timeLeft)
      return toAct

    # ACTION SENARIO 5: explore food or capsule
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    bestAction = random.choice(bestActions)

    # ACTION SENARIO 6: about to win
    foodLeft = len(self.getFood(gameState).asList())
    if foodLeft <= 2:
      values = [self.evaluateGoHome(gameState, a) for a in actions]
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      bestAction = random.choice(bestActions)
      #print('=== 2300 ===', self.index, bestAction, timeLeft)
      return bestAction
    toAct = random.choice(bestActions)
    #print('=== 326 ===', self.index, toAct, timeLeft)
    return toAct
    toAct = random.choice(bestActions)
    #print('=== 326 ===', self.index, toAct, timeLeft)
    return toAct
  
  def reachedImpasse(self, gameState, myPos):
    inImpasseRegion = bool(myPos in self.frontierPoints)
    #print('=== 320 ===', friendPos, myPos, gstPos)
    gstPos = self.checkStateSafe(gameState)
    return inImpasseRegion and gstPos
  
  def evaluateGoHome(self, gameState, action):
    features = self.getFeaturesGoHome(gameState, action)
    weights = self.getWeightsGoHome(gameState, action)
    return features * weights

  def getFeaturesGoHome(self, gameState, action):
    features = util.Counter()
    myPos = gameState.getAgentPosition(self.index)
    ghsPosition = self.checkStateSafe(gameState)
    successor = gameState.generateSuccessor(self.index, action)
    nextPos = successor.getAgentState(self.index).getPosition()
    if ghsPosition:
      for _, pos in ghsPosition:
        features['distToGhost'] += self.getMazeDistance(pos, nextPos)
    else: 
      features['distToGhost'] = 0
    if self.distToHome[myPos]:
      features['homeDist'] = -1*self.distToHome[myPos][0]
    else:
      features['homeDist'] = 0

    # is eaten
    if ghsPosition:
      if self.mayBeEaten(nextPos, ghsPosition):
        features['isEaten'] = 1
      else:
        features['isEaten'] = 0

    return features

  def getWeightsGoHome(self,gameState, action):
    return {'homeDist': 30, 'distToGhost': 15, 'isEaten': -80}

  def evaluateImpasse(self, gameState, action):
    features = self.getFeaturesImpasse(gameState, action)
    weights = self.getWeightsImpasse(gameState, action)
    return features * weights

  def getFeaturesImpasse(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    nextState = successor.getAgentState(self.index)
    foodList = self.getFood(successor).asList()
    nextPos = successor.getAgentState(self.index).getPosition()
    myPos = gameState.getAgentPosition(self.index)

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getFoodDistance(nextPos, food, gameState) for food in foodList])
      features['distanceToFood'] = minDistance

    # Away from ghost
    ghsPosition = self.checkStateSafe(gameState)
    if ghsPosition:
      for enemyIdx, pos in ghsPosition:
        #print('=== 361 ===', self.belief)
        self.updateBelief(pos, enemyIdx)
        #print(self.belief)
        features['distToGhost'] += self.getMazeDistance(pos, nextPos)
    
    features['distToGhost'] += self.getApproxGhostDistance(nextPos)
    #print(self.belief)

    # penalise stop
    if action == Directions.STOP: features['stop'] = 1

    # penalise reverse
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    # invader
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if len(invaders) > 0:
      dists = [self.getMazeDistance(nextPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    # invader number
    features['numInvaders'] = len(invaders)

    # isPacman
    features['isPacman'] = successor.getAgentState(self.index).isPacman

    # is eaten
    if ghsPosition:
      if self.mayBeEaten(nextPos, ghsPosition):
        features['isEaten'] = 1
      else:
        features['isEaten'] = 0
    return features

  def getApproxGhostDistance(self, nextPos):
    dist = 0
    for idx in self.enemies:

      if len(self.belief[idx].keys()):
        for pos, belief in self.belief[idx].items():
          dist += belief*self.getMazeDistance(pos, nextPos)
    return dist

  def updateBelief(self, position, idx):
    alreadyExistedPositions = self.belief[idx].keys()
    if position in alreadyExistedPositions or (not alreadyExistedPositions):
      possiblePositions = [(position[0]+i, position[1]+j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (position[0]+i, position[1]+j) in self.legalPositions]
      for pos in possiblePositions:
        self.belief[idx][pos] += 1/9
      self.belief[idx][position] += 1/9
    else:
      self.belief[idx] = util.Counter()    # if out of threat, clean our belief

  def getDistToFriend(self, friendPos, myPos):
    favoredY = self.favoredY
    myDist = -abs(myPos[1] - favoredY)
    friendDist = self.getMazeDistance(myPos, friendPos)
    if friendDist <= 4:
      return friendDist + favoredY
    return favoredY

  def mayBeEaten(self, nextPos, gstPos):
    beEaten = 0
    for _, pos in gstPos:
      gstNextPos = [(pos[0]+i, pos[1]+j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
      if nextPos in gstNextPos:
        beEaten = 1
    return beEaten

  def getWeightsImpasse(self, gameState, action):
    return {'distanceToFood': -2.5, 'distToGhost': 20,\
    'stop': -15, 'reverse': -5, 'invaderDistance': -3, \
    'numInvaders': -3, 'isPacman': 3, 'isEaten': -80, 'distToFriend': 40}

  def getEvaluation(self, gameState, allIndex):
    myPos = gameState.getAgentPosition(allIndex[0])
    enermyPos = gameState.getAgentPosition(allIndex[1])
    value = self.getMazeDistance(myPos, enermyPos)

    if myPos == self.start:
      value -= 200
    return value

  def max2(self, gameState, depth, playerIndex, allIndexes):
    if depth == 0 or gameState.getLegalActions(playerIndex) == None:
      return (self.getEvaluation(gameState, allIndexes), None)
    else:
      #bestActionValue = -9999
      
      actions = gameState.getLegalActions(playerIndex)
      #value = util.Counter()
      actionValues = []
      applicableActions = []
      myPosList = []
      for action in actions:
        # Avoid return "STOP" in the 
        
        if action!= "STOP":
          successor = gameState.generateSuccessor(playerIndex, action)
          enermyIndex = allIndexes[1]
          actionValue,a = self.min2(successor, depth-1,enermyIndex, allIndexes)
          myPos = successor.getAgentPosition(playerIndex)
          myPosList.append(myPos)
          actionValues.append(actionValue)
          applicableActions.append(action)
      
      # when the final action (depth = self.minimaxDepth) list is empty, return 'STOP'.
      if len(applicableActions) == 0:
        return 0, 'STOP'

      maxValue = max(actionValues)
      bestActions = [a for a, v, p in zip(applicableActions, actionValues,myPosList) if v == maxValue]
      bestActionsPos = [p for a, v, p in zip(applicableActions, actionValues,myPosList) if v == maxValue]

      if depth!= self.miniMaxDepth:
        return maxValue, random.choice(bestActions)

      # select next action based on features (try to avoid go to the dead end)
      if depth == self.miniMaxDepth:
        #closestToHome = []
        selectedActionsAtHome = []
        distToHomeList = []
        selectedActionsInEnermy = []
        actionsInEnermy = []
        for bestAction, pos in zip(bestActions, bestActionsPos):
          if pos not in self.enemyCells:
            selectedActionsAtHome.append(bestAction)
          else:
            distToHomeList.append(self.distToHome[pos][0])
            actionsInEnermy.append(bestAction)
        if len(distToHomeList) !=0:
          closestToHomeDist = min(distToHomeList)
          selectedActionsInEnermy = [a for a,d in zip(actionsInEnermy, distToHomeList) if d == closestToHomeDist]

          selectedActions = selectedActionsAtHome + selectedActionsInEnermy
        else:
          selectedActions = selectedActionsAtHome 
        if len(selectedActions) != 0:
          #print('407--', random.choice(selectedActions))
          return 0, random.choice(selectedActions)
        #print('408--', random.choice(selectedActions))
        return 0, random.choice(bestActions)
            
        #suuceeor valid actions
        
  def min2(self, gameState, depth, playerIndex, allIndexes):
    bestActionValue = 9999
    bestAction = None
    actions = gameState.getLegalActions(playerIndex)
    #value = util.Counter()
    for action in actions:
      if action!= "STOP":
        successor = gameState.generateSuccessor(playerIndex, action)
        nextIndex = allIndexes[1]
        actionValue,a = self.max2(successor, depth-1, nextIndex, allIndexes)
        #print("369++++",depth-1, actionValue, action,a)
        if bestActionValue > actionValue:
          bestAction = action
          bestActionValue = actionValue
    return bestActionValue, bestAction
  

  def isCloseToFood(self, gameState, actions):
    foodNum = len(self.getFood(gameState).asList())
    isClose = 0
    for action in actions:
      successor = gameState.generateSuccessor(self.index, action)
      foodNumNew = len(self.getFood(successor).asList())
      if foodNum > foodNumNew:
        isClose = 1
    return isClose

  def evaluatePatrol(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeaturesPatrol(gameState, action)
    weights = self.getWeightsPatrol(gameState, action)
    return features * weights

  def getFeaturesPatrol(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    nextPos = successor.getAgentPosition(self.index)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)

    patrolArea = self.frontierPoints[int(len(self.frontierPoints)/2):]
    features['distToPatrol'] = self.getDistToPatrol(myPos, patrolArea)

    # invader
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if len(invaders) > 0:
      dists = [self.getMazeDistance(nextPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
    return features
  
  def getDistToPatrol(self, myPos, patrolArea):
    dists = 0
    i = 0
    for pos in patrolArea:
      dists += (self.getMazeDistance(pos, myPos))
      i += 1
      
    return dists/i

  def getWeightsPatrol(self, gameState, action):
    return {'numInvaders': -70, 'onDefense': 100, 'distToPatrol': -20, 'invaderDistance': -8}

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    nextState = successor.getAgentState(self.index)
    foodList = self.getFood(successor).asList()
    nextPos = successor.getAgentState(self.index).getPosition()

    # score  
    features['successorScore'] = -len(foodList)

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getFoodDistance(nextPos, food, gameState) for food in foodList])
      features['distanceToFood'] = minDistance

    # Distance to Power Capsule
    capsule = self.getCapsules(gameState)
    if capsule:
      capsuleDist = min([self.getMazeDistance(cap, nextPos) for cap in capsule])
      features['distanceToCapsule'] = capsuleDist
    else:
      features['distanceToCapsule'] = 0 # since eaten

    # Away from ghost
    ghsPosition = self.checkStateSafe(gameState)
    if ghsPosition:
      for _, pos in ghsPosition:
        features['distToGhost'] += self.getMazeDistance(pos, nextPos)
    else: 
      features['distToGhost'] = 0

    # need to cash in
    features['cashIn'] = self.needToCashIn(nextPos, nextState, self.minPelletsToCashIn)

    # penalise stop
    if action == Directions.STOP: features['stop'] = 1

    # penalise reverse
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    # invader
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if len(invaders) > 0:
      dists = [self.getMazeDistance(nextPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    # invader number
    features['numInvaders'] = len(invaders)

    # isPacman
    features['isPacman'] = successor.getAgentState(self.index).isPacman

    # is eaten
    if ghsPosition:
      if self.mayBeEaten(nextPos, ghsPosition):
        features['isEaten'] = 1
      else:
        features['isEaten'] = 0

    return features

  def getFoodDistance(self, myPos, food, gameState):
    """
    Force one agent to eat top food
    """
    favoredY = self.favoredY
    return self.getMazeDistance(myPos, food) + abs(favoredY - food[1])

  def needToCashIn(self, myPos, nextState, maxCarry):
    # if we have enough pellets, attempt to cash in
    if nextState.numCarrying >= maxCarry:
      return 1
    else:
      return 0

  def getWeights(self, gameState, action):
    return {'successorScore': 80, 'distanceToFood': -1.8, \
    'distanceToCapsule': -5, 'distToGhost': 20, 'cashIn': 10, \
    'stop': -12, 'reverse': -2, 'invaderDistance': -1, \
    'numInvaders': -2, 'isPacman': 3, 'isEaten': -40}

  def checkStateSafe(self, gameState):
    """
    Check if current state may be threatened by
    Returns ghost position, if none return None
    """
    enemy = self.getEnemy(gameState)
    enemyGhost = enemy['Ghost']
    agentPos = gameState.getAgentPosition(self.index)

    # check if ghost scared
    minDist = 9999
    for index, pos in enemyGhost:
      scared = gameState.data.agentStates[index].scaredTimer
      if scared > 2:
        enemyGhost = None
      dist = self.getMazeDistance(agentPos, pos)

      if dist < minDist:
        minDist = dist
    if minDist > 4:
      enemyGhost = None

    if not enemyGhost:
      return None
    return enemyGhost
    
  def getEnemy(self, gameState): 
    """
    Returns the enemy state as a dictionary
    """
    enemyState = {'Pacman': [], 'Ghost':[]} 
    enemy = gameState.getBlueTeamIndices() if gameState.isOnRedTeam(self.index) else gameState.getRedTeamIndices()
    for index in enemy:
      eState = gameState.data.agentStates[index]
      if eState.isPacman and (eState.getPosition() != None):
        enemyState['Pacman'].append((index, eState.getPosition()))
      elif (not eState.isPacman) and (eState.getPosition() != None):
        enemyState['Ghost'].append((index, eState.getPosition()))
    return enemyState

  def escape(self, actions, gameState):
    """
    Find an escape action
    """
    vals = []
    for action in actions:
      successor = self.getSuccessor(gameState, action)
      nextPos = successor.getAgentPosition(self.index)
      features = self.getFeaturesEscape(gameState, nextPos)
      weights = self.getWeightsEscape(gameState)
      vals.append(features * weights)

    maxValue = max(vals)
    bestActions = [act for act, val in zip(actions, vals) if val == maxValue]

    return random.choice(bestActions)

  def getFeaturesEscape(self, gameState, nextPos):
    features = util.Counter()
    foodList = self.getFood(gameState).asList()
    myPos = gameState.getAgentPosition(self.index)
    features['successorScore'] = -len(foodList)
    ghsPosition = self.checkStateSafe(gameState)

    if ghsPosition:
      for _, pos in ghsPosition:
        features['distToGhost'] += self.getMazeDistance(pos, nextPos)
    else: 
      features['distToGhost'] = 0
    if self.distToHome[myPos]:
      features['toHome'] = -1*self.distToHome[myPos][0]
    else:
      features['toHome'] = 0

    # is eaten
    if ghsPosition:
      if self.mayBeEaten(nextPos, ghsPosition):
        features['isEaten'] = 1
      else:
        features['isEaten'] = 0

    return features

  def getWeightsEscape(self, gameState):
    return {'successorScore': 2, 'distToGhost': 40, 'toHome': 10, 'isEaten': -40}

class DefensiveReflexAgent(ClassicPlanAgent):
  """
  A defensive agent that keeps its side Pacman-free.
  With belief of where the pacman may be.
  """
  
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # INIT VARIABLES TO USE
    gstPos = self.checkStateSafe(gameState)
    myPos = gameState.getAgentPosition(self.index)
    Pacman = gameState.getAgentState(self.index).isPacman
    myState = gameState.getAgentState(self.index)
    defendFood = self.getFoodYouAreDefending(gameState).asList()
    timeLeft = gameState.data.timeleft

    # === ABOUT TO LOSE SENARIO ===
    if len(defendFood) <= 4:
      values = [self.evaluatePatrol(gameState, a) for a in actions]
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      bestAction = random.choice(bestActions)
      #print('=== 767 ===', self.index, bestAction, timeLeft)
      return bestAction
    
    # === OFFENSIVE SENARIO ===
    if len(defendFood) > 10 and timeLeft > 80:

      # CASE 1: REACHED-AN-IMPASSE SENARIO ===
      isInImpasse = self.reachedImpasse(gameState, myPos)
      if isInImpasse:
        #print('IMPASSE DETECTED', myPos, gstPos)
        values = [self.evaluateImpasse(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        bestAction = random.choice(bestActions)
        #print('=== 781 ===', self.index, bestAction, timeLeft)
        return bestAction

      # CASE 2: carrying enough food, go home
      goHome = self.needToCashIn(myPos, myState, self.minPelletsToCashIn)
      isHome = self.distToHome[myPos]
      isCloseToFood = self.isCloseToFood(gameState, actions)
      if goHome and (isHome) and (not isCloseToFood):
        dist, end = self.distToHome[myPos]
        # BFS find shortest path to home
        _, path = self.getBfsPath(gameState, end, myPos)
        toAct = path.pop(0)

        # check if threatened by ghost
        threatToHome = self.checkStateSafe(gameState)
        Pacman = gameState.getAgentState(self.index).isPacman
        isHome = (self.distToHome[myPos])
        if threatToHome and isHome:    # if threatened and not at home
          # escape
          if threatToHome and Pacman:
            enermyIndex = [tup[0] for tup in gstPos]
            if len(enermyIndex) == 1:
              allIndexes = [self.index, enermyIndex[0]]
              depth = self.miniMaxDepth
              v, toAct = self.max2(gameState, depth, self.index, allIndexes)
              # new minimax 2 
            else:
              #toAct = self.getMiniMaxAction(gameState, myPos, enemyPos)
              toAct = random.choice(actions)
        #print('=== 810 ===', self.index, toAct, timeLeft)
        return toAct
      
      # CASE 3: while eating, threatened
      threatToHome = self.checkStateSafe(gameState)
      isHome = (self.distToHome[myPos])
      if threatToHome and (isHome):
        dist, home = self.distToHome[myPos]
        if dist < 4:    # is close to home, 
          _, actionSeq = self.getBfsPath(gameState, home, myPos)
          toAct = actionSeq.pop(0)
          actedPos = (gameState.generateSuccessor(self.index, toAct)).getAgentPosition(self.index)
          for _, gstPos in threatToHome:
            if self.getMazeDistance(actedPos, gstPos) < 2:
              toAct = self.escape(actions, gameState)
        else:
          toAct = self.escape(actions, gameState)
        #print('=== 827 ===', self.index, toAct, timeLeft)
        return toAct

      # CASE 3: no threats and still hungry  
      values = [self.evaluate(gameState, a) for a in actions]

    # === DEFENDIVE SENARIO ===
    else:
      values = [self.evaluateDefensive(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())
    
    # === ABOUT-TO-WIN SENARIO ===
    if foodLeft <= 2:
      values = [self.evaluateGoHome(gameState, a) for a in actions]
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      bestAction = random.choice(bestActions)
      #print('=== 2300 ===', self.index, bestAction, timeLeft)
      return bestAction
    toAct = random.choice(bestActions)
    #print('=== 326 ===', self.index, toAct, timeLeft)
    return toAct
    bestAction = random.choice(bestActions)
    #print('=== 873 ===', self.index, bestAction, timeLeft)
    return bestAction
    
  def getWeightsImpasse(self, gameState, action):
    # Give more incentive to intercept enemy pacman
    return {'distanceToFood': -2, 'distToGhost': 22,\
    'stop': -12, 'reverse': -3, 'invaderDistance': -4, \
    'numInvaders': -6, 'isPacman': 3, 'isEaten': -80, 'distToFriend': 40}

  def evaluatePatrol(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeaturesPatrol(gameState, action)
    weights = self.getWeightsPatrol(gameState, action)
    return features * weights

  def getFeaturesPatrol(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)

    patrolArea = self.frontierPoints[:int(len(self.frontierPoints)/2)]
    features['distToPatrol'] = self.getDistToPatrol(myPos, patrolArea)
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -2.5, \
    'distanceToCapsule': -2, 'distToGhost': 30, 'cashIn': 0, \
    'stop': -15, 'reverse': -2, 'invaderDistance': -3.5, \
    'numInvaders': -3.5, 'isPacman': 3}
  
  def getFoodDistance(self, myPos, food, gameState):
    favoredY = abs(self.height-self.favoredY)
    return self.getMazeDistance(myPos, food) + abs(favoredY - food[1])

  def getDistToFriend(self, friendPos, myPos):
    favoredY = abs(self.height-self.favoredY)
    myDist = -abs(myPos[1] - favoredY)
    friendDist = self.getMazeDistance(myPos, friendPos)
    if friendDist <= 4:
      return friendDist + favoredY
    return favoredY

  ##########################################
  #      BELOW IS FOR DEFENSIVE STATE      #
  ##########################################

  def getFeaturesDefensive(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    invaderPos = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
    if invaderPos:
      for inv in invaderPos:
        features['invaderDistToHome'] += -1/self.getInvaderDistToHome(inv)
    else:
      features['invaderDistToHome'] = 0

    features['numInvaders'] = len(invaders)

    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    # dist to pacman
    pacmanProbIn = self.beliefInPacmanPosition(gameState)
    features['ToFoodCluster'] = self.getMazeDistance(pacmanProbIn, myPos)

    return features
  
  def defendTheCluster(self, gameState):
    foodList = self.getFoodYouAreDefending(gameState).asList()
    bigClusterFood = []
    for food1 in foodList:
      clusterSize = 0
      for food2 in foodList:
        if self.getMazeDistance(food1, food2) <= 3:
          clusterSize += 1
      bigClusterFood.append((food1, clusterSize))
    
    return bigClusterFood

  def beliefInPacmanPosition(self, gameState):
    """
    Returns a position that our ghost should patrol
    """
    enemyDist = []
    positionToPatrol = []
    myPos = gameState.getAgentPosition(self.index)
    enemyPacmanPosition = self.getEnemy(gameState)['Pacman']
    cluster = self.defendTheCluster(gameState)
    minDist = 999
    if enemyPacmanPosition:
      for idx, pos in enemyPacmanPosition:
        curDist = self.getMazeDistance(pos, myPos)
        if minDist > curDist:
          minDist = curDist
          minPos = pos
      return minPos

    else:
      allDist = gameState.getAgentDistances()
      enemyDist = []
      for idx in self.enemies:
        enemyDist.append(allDist[idx])

    myPos = gameState.getAgentPosition(self.index)
    for c, size in cluster:
      for dist in enemyDist:
        CI = range(dist-4, dist+4)
        if self.getMazeDistance(c, myPos) in CI:
          positionToPatrol.append((c, size))
    if positionToPatrol:
      patrol = sorted(positionToPatrol, key = lambda x: x[1])[-1]

    else:
      patrol = sorted(cluster, key = lambda x: x[1])[-1]
      #('=== 596 ===', patrol[0])
    return patrol[0]


  def getInvaderDistToHome(self, invaderPos):
    dist = [self.getMazeDistance(enemyhome, invaderPos) for enemyhome in self.frontierPointsEnemy]
    return sum(dist)/len(dist)

  def evaluateDefensive(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeaturesDefensive(gameState, action)
    weights = self.getWeightsDefensive(gameState, action)
    return features * weights

  def getSuccessorDefensive(self, gameState, action):
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

  def getWeightsDefensive(self, gameState, action):
    return {'numInvaders': -70, 'onDefense': 100, 'invaderDistToHome': 30, 'invaderDistance': -20, \
    'stop': -100, 'reverse': -3, 'ToFoodCluster': -15}
