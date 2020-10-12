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
    self.capsulState, self.actionCapsule = self.getBfsPath(self.frontierState, self.capsulePosition, self.closestFrontier)
    # minimax initial set up
    # face 1 enermy depth
    self.miniMaxDepth = 4
    # face 2 enermy depth
    self.miniMaxDepth2 = 3
    # Initialisation for hyper-parameters
    self.epsilon = 0.75

    # enemies
    self.enemies = self.getOpponents(gameState)
    self.beliefs = {}    # Initialize the belief to 1
    
    for enemy in self.enemies:
      self.beliefs[enemy] = util.Counter()
      self.beliefs[enemy][gameState.getAgentPosition(enemy)] = 1    # beliefs for each agent


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

    # === ABOUT TO LOSE SENARIO ===
    defendFood = self.getFoodYouAreDefending(gameState).asList()
    if len(defendFood) <= 4:
      
      values = [self.evaluatePatrol(gameState, a) for a in actions]
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      return random.choice(bestActions)


    # ACTION STEP 1: reach frontier first
    if self.actionFrontier:
      toAct = self.actionFrontier.pop(0)
      return toAct

    # ACTION STEP 2: after reaching frontier
    if self.actionCapsule:
      toAct = self.actionCapsule.pop(0)

      # check if planned step is safe
      gstPos = self.checkStateSafe(gameState)
      if not gstPos:    # safe, ghost not detected
        return toAct

      # otherwise, ghost detected, run.
      self.actionCapsule = []    # ditch all pre-planed path

    # ACTION SENARIO 3: if detects ghost, use minimax to get out of the way
    gstPos = self.checkStateSafe(gameState)
    myPos = gameState.getAgentPosition(self.index)
    Pacman = gameState.getAgentState(self.index).isPacman

    if gstPos and Pacman:
      
      
      enermyIndex = [tup[0] for tup in gstPos]
      depth = self.miniMaxDepth
      toAct = self.minimax(gameState, depth, self.index, enermyIndex)
      
      
      #if len(enermyIndex) == 1:
        #print("BEGIN MINIMAX++++")
        #allIndexes = [self.index, enermyIndex[0]]
        #depth = self.miniMaxDepth
        #v, toAct = self.max2(gameState, depth, self.index, allIndexes)
        #print("+++MINIMAX RESULT", toAct, v)
        # new minimax 2 
     
      return toAct

    
      

    # ACTION SENARIO 4: if food more than threshold, need to cash in
    myState = gameState.getAgentState(self.index)

    goHome = self.needToCashIn(myPos, myState, self.minPelletsToCashIn)
    isHome = self.distToHome[myPos]
    isCloseToFood = self.isCloseToFood(gameState, actions)
    if goHome and (isHome) and (not isCloseToFood):
      dist, end = self.distToHome[myPos]
      _, path = self.getBfsPath(gameState, end, myPos)
      toAct = path.pop(0)
      return toAct

    # ACTION SENARIO 5: explore food or capsule
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    bestAction = random.choice(bestActions)

    foodLeft = len(self.getFood(gameState).asList())
    
    # ACTION SENARIO 6: about to win
    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        gstPos = self.checkStateSafe(successor)

        if dist < bestDist and (not gstPos): # no ghost and closer
          bestAction = action
          bestDist = dist

        elif (not gstPos):
          bestAction = action
          bestDist = dist

        else:
          bestAction = self.escape(actions, gameState)

      return bestAction

    return random.choice(bestActions)

  def minimax(self, gameState, depth, playerIndex, enermyIndex):
    if len(enermyIndex) == 1:
      allIndexes = [self.index, enermyIndex[0]]
      depth = self.miniMaxDepth
      v, toAct = self.max2(gameState, depth, self.index, allIndexes)
      #print("+++1 Enermy MINIMAX RESULT", toAct, v)
    elif len(enermyIndex) == 2:
      #actions = gameState.getLegalActions(self.index)
      allIndexes  = [playerIndex] + enermyIndex
      depth = self.miniMaxDepth2
      print("\n \n")
      print("+++ Enermy MINIMAX BEGIN+++326")
      v, toAct = self.maxn(gameState, depth, playerIndex, allIndexes)
      #toAct = random.choice(actions)
      #print("HAVNT IMPLEMENTED")
      print("+++2 Enermy MINIMAX RESULT", toAct, v)
    return toAct

  def getEvaluation(self, gameState, allIndex):
    #print(len(allIndex))
    if len(allIndex) == 2:
      myPos = gameState.getAgentPosition(allIndex[0])
      enermyPos = gameState.getAgentPosition(allIndex[1])
      value = self.getMazeDistance(myPos, enermyPos)
      if myPos == self.start:
        value -= 200
      return value
    else:
      myPos = gameState.getAgentPosition(allIndex[0])
      enermyPos1 = gameState.getAgentPosition(allIndex[1])
      enermyPos2 = gameState.getAgentPosition(allIndex[2])
      distToEnermy1 = self.getMazeDistance(myPos, enermyPos1)
      distToEnermy2 = self.getMazeDistance(myPos, enermyPos2)
      minDistToEnermy = min(distToEnermy1, distToEnermy2)

      #print('=== 239 === ', myPos,enermyPos)
      return [(minDistToEnermy, -distToEnermy1, -distToEnermy2)]
    
    

  def max2(self, gameState, depth, playerIndex, allGameIndexes):
    if depth == 0 or gameState.getLegalActions(playerIndex) == None:
      return (self.getEvaluation(gameState, allGameIndexes), None)
    else:
      #bestActionValue = -9999
      
      actions = gameState.getLegalActions(playerIndex)
      #value = util.Counter()
      actionValues = []
      applicableActions = []
      myPosList = []
      for action in actions:
        # Avoid return "STOP" in the 
        #if (depth == self.miniMaxDepth and action!= "STOP") or depth!= self.miniMaxDepth:
        if action!= "Stop":
          successor = gameState.generateSuccessor(playerIndex, action)
          enermyIndex = allGameIndexes[1]
          actionValue,a = self.min2(successor, depth-1,enermyIndex, allGameIndexes)
          myPos = successor.getAgentPosition(playerIndex)
          myPosList.append(myPos)
          #print("352++++",depth-1, actionValue, action, a)
          actionValues.append(actionValue)
          applicableActions.append(action)
      
      # when the final action (depth = self.minimaxDepth) list is empty, return 'STOP'.
      if len(applicableActions) == 0:
        return 0, 'Stop'

      maxValue = max(actionValues)
      bestActions = [a for a, v, p in zip(applicableActions, actionValues,myPosList) if v == maxValue]
      bestActionsPos = [p for a, v, p in zip(applicableActions, actionValues,myPosList) if v == maxValue]

      if depth!= self.miniMaxDepth:
        #print("384", random.choice(bestActions))
        return maxValue, random.choice(bestActions)

      # select next action based on features (try to avoid go to the dead end)
      if depth == self.miniMaxDepth:
        #closestToHome = []
        selectedActions = self.selectMiniMaxAction(bestActions, bestActionsPos, gameState)
        return 0, random.choice(selectedActions)
  
  # detect the number of walls surround a position.
  def detectSurroundWall(self, gameState, pos):
    wallNum = 0
    if gameState.hasWall(pos[0] + 1, pos[1]):
      wallNum += 1
    if gameState.hasWall(pos[0] - 1, pos[1]):
      wallNum += 1
    if gameState.hasWall(pos[0], pos[1] + 1):
      wallNum += 1
    if gameState.hasWall(pos[0] + 1, pos[1] - 1):
      wallNum += 1
    return wallNum
    

              
  # Select actions with shortest distance to home from the highest depth   
  def selectMiniMaxAction(self, bestActions, bestActionsPos, gameState):
          selectedActionsAtHome = []
          distToHomeList = []
          selectedActionsInEnermy = []
          actionsInEnermy = []
          for bestAction, pos in zip(bestActions, bestActionsPos):
            if pos not in self.enemyCells:
              selectedActionsAtHome.append(bestAction)
            elif pos in self.enemyCells and self.detectSurroundWall(gameState, pos) != 3:
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
            return selectedActions
            #print('408--', random.choice(selectedActions))
          else:
            return bestActions


      


  def min2(self, gameState, depth, playerIndex, allIndexes):
    bestActionValue = 9999
    bestAction = None
    actions = gameState.getLegalActions(playerIndex)
    #value = util.Counter()
    for action in actions:
      if action!= 'Stop':
        successor = gameState.generateSuccessor(playerIndex, action)
        nextIndex = allIndexes[1]
        actionValue,a = self.max2(successor, depth-1, nextIndex, allIndexes)
        #print("369++++",depth-1, actionValue, action,a)
        if bestActionValue > actionValue:
          bestAction = action
          bestActionValue = actionValue
    return bestActionValue, bestAction
  
  
  def maxn(self, gameState, depth, playerIndex, allIndexes):
    if depth == 0 or gameState.getLegalActions(playerIndex) == None or gameState.isOver():
      return (self.getEvaluation(gameState, allIndexes), None)
    else:
      #bestActionValue = -9999
      playerIndexInList = allIndexes.index(playerIndex)
      actions = gameState.getLegalActions(playerIndex)
      #value = util.Counter()
      actionValues = []
      applicableActions = []
      myPosList = []
      for action in actions:
        # Avoid return "STOP" in the 
        #if (depth == self.miniMaxDepth and action!= "STOP") or depth!= self.miniMaxDepth:
        if action!= 'Stop':
          successor = gameState.generateSuccessor(playerIndex, action)
          playerInallIndex = (allIndexes.index(playerIndex) + 1)%len(allIndexes)
          enermyIndex = allIndexes[playerInallIndex]
          actionValue,a = self.maxn(successor, depth-1,enermyIndex, allIndexes)
          myPos = successor.getAgentPosition(playerIndex)
          for i in range(len(actionValue)):
            actionValues.append(actionValue[i])
            applicableActions.append(action)
          
            myPosList.append(myPos)
          #print("460++++",depth-1, actionValue, action, a)
          #actionValues = actionValues + actionValue
          
          
      
      # when the final action (depth = self.minimaxDepth) list is empty, return 'STOP'.
      if len(applicableActions) == 0:
        print('=================================================================')
        return [(0,0,0)], 'Stop'
      #print('494',actionValues, [playerIndexInList])
      #print('495',[valueTuple for valueTuple in actionValues])
      maxPlayerValue = max([valueTuple[playerIndexInList] for valueTuple in actionValues])
      
      #maxValue = max(actionValues)
      bestActionValueTuples = [v for a, v, p in zip(applicableActions, actionValues,myPosList) if v[playerIndexInList] == maxPlayerValue]
      bestActions = [a for a, v, p in zip(applicableActions, actionValues,myPosList) if v[playerIndexInList] == maxPlayerValue]
      bestActionsPos = [p for a, v, p in zip(applicableActions, actionValues,myPosList) if v[playerIndexInList] == maxPlayerValue]
      #print('489 maxPlayer Value',depth, maxPlayerValue, bestActions, bestActionValueTuples,actionValues)
      if depth!= self.miniMaxDepth2:
        toAct = random.choice(bestActions)
        #print("474", toAct,bestActionValueTuples)
        return bestActionValueTuples, toAct

      # select next action based on features (try to avoid go to the dead end)
      if depth == self.miniMaxDepth2:
        #closestToHome = []
        selectedActions = self.selectMiniMaxAction(bestActions, bestActionsPos, gameState)
        toAct = random.choice(selectedActions)

        return [(0,0,0)], toAct

  
  
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
    return {'numInvaders': -70, 'onDefense': 100, 'distToPatrol': -20, 'invaderDistance': -2}

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
    if self.start == nextPos:
      features['isEaten'] = 1
    else:
      features['isEaten'] = 0

    return features

  def getFoodDistance(self, myPos, food, gameState):
    """
    Force one agent to eat top food
    """
    favoredY = gameState.data.layout.height
    return self.getMazeDistance(myPos, food) + abs(favoredY - food[1])

  def needToCashIn(self, myPos, nextState, maxCarry):
    # if we have enough pellets, attempt to cash in
    if nextState.numCarrying >= maxCarry:
      return 1
    else:
      return 0

  def getWeights(self, gameState, action):
    return {'successorScore': 80, 'distanceToFood': -1.5, \
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
      #print('=== 492 ===', features, '\n', weights)
      vals.append(features * weights)

    maxValue = max(vals)
    bestActions = [act for act, val in zip(actions, vals) if val == maxValue]

    return random.choice(bestActions)

  def getFeaturesEscape(self, gameState, nextPos):
    features = util.Counter()
    #print('CALLING ESCAPE FUNCTION')
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
    if self.start == nextPos:
      features['isEaten'] = 1
    else:
      features['isEaten'] = 0

    return features

  def getWeightsEscape(self, gameState):
    return {'successorScore': 2, 'distToGhost': 40, 'toHome': 10, 'isEaten': -40}

class DefensiveReflexAgent(ClassicPlanAgent):
  """
  A defensive agent that keeps its side Pacman-free.
  With belief of where the pacman may be
  """
  
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    myState = gameState.getAgentState(self.index)
    myPos = gameState.getAgentPosition(self.index)

    defendFood = self.getFoodYouAreDefending(gameState).asList()

    timeLeft = gameState.data.timeleft

    # === ABOUT TO LOSE SENARIO ===
    if len(defendFood) <= 4:
      values = [self.evaluatePatrol(gameState, a) for a in actions]
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      return random.choice(bestActions)
    
    # === OFFENSIVE SENARIO ===
    if len(defendFood) > 10 and timeLeft > 80:
      # CASE 1: carrying enough food, go home
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
        isHome = (self.distToHome[myPos])
        if threatToHome and isHome:    # if threatened and not at home
          # escape
          toAct = self.escape(actions, gameState)
        return toAct
      
      # CASE 2: while eating, threatened
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
        return toAct

      # CASE 3: no threats and not still hungry  
      values = [self.evaluate(gameState, a) for a in actions]

    # === DEFENDIVE SENARIO ===
    else:
      values = [self.evaluateDefensive(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())
    
    # === ABOUT-TO-WIN SENARIO ===
    if foodLeft <= 2:
      bestDist = 9999

      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        gstPos = self.checkStateSafe(successor)

        if dist < bestDist and (not gstPos): # no ghost and closer
          bestAction = action
          bestDist = dist

        elif (not gstPos):
          bestAction = action
          bestDist = dist

        else:
          bestAction = self.escape(actions, gameState)

      return bestAction
    return random.choice(bestActions)
    

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
    return {'successorScore': 120, 'distanceToFood': -1, \
    'distanceToCapsule': -2, 'distToGhost': 30, 'cashIn': 0, \
    'stop': -15, 'reverse': -2, 'invaderDistance': -2, \
    'numInvaders': -3, 'isPacman': 3}
  
  def getFoodDistance(self, myPos, food, gameState):
    favoredY = 0.0
    return self.getMazeDistance(myPos, food) + abs(favoredY - food[1])

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
    #print('=== 545 ===', pacmanProbIn, myPos)
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
        #print('=== 575 ===', enemyPacmanPosition, pos)
        curDist = self.getMazeDistance(pos, myPos)
        if minDist > curDist:
          minDist = curDist
          minPos = pos
          #print('=== 580 ===', pos)
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
