# myTeam.py
# version 2.3 fix bug
# Time; 10/17/2020 
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

# global variable
belief = {}

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
    self.minPelletsToCashIn = int(self.totalFoodNum*0.35)

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

    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] >= 1]
    self.myCells = [c for c in self.legalPositions if c not in self.enemyCells]
    
    # A dictionary for the closest distance from any enemy cell to home
    self.distToHome = self.getDistToHome(self.frontierPoints, self.enemyCells)
    # A sequence of action to the closest enemy cell
    self.frontierState, self.actionFrontier = self.toFrontier(gameState, self.frontierPoints, self.start)

    self.closestFrontier = self.frontierState.getAgentPosition(self.index)
    if self.closestFrontier in self.frontierPoints[:int(len(self.frontierPoints)/2)]:
      self.favoredY = 0.0
    else:
      self.favoredY = self.height

    # minimax initial set up
    # face 1 enemy depth
    self.miniMaxDepth = 4
    # face 2 enemy depth
    self.miniMaxDepth2 = 3
    # Initialisation for hyper-parameters
    self.epsilon = 0.75

    # enemies
    self.enemies = self.getOpponents(gameState)
    global belief    # inference on ghost position
    self.trackPosition = {}   # track historical positions the enemy has been to
    
    self.findDeadEnd(gameState)
    #print("===Find dead end poses=== 145\n", "\n",self.deadEndPoses)
    #print(gameState.data.layout)
    fakeLay = gameState.data.layout.deepCopy()
    #print("FAKE\n",fakeLay)
    maxY = self.height - 1
    newlay = ""
    for y in range(self.height):
      for x in range(self.width):
        if (x, maxY - y) in self.deadEndPoses:
          newlay +='X'
        else:
          newlay += fakeLay.layoutText[y][x]
        

        #layoutChar = fakeLay.layoutText[maxY - y][x]
        #print(layoutChar)
      newlay += '\n'
    """for (x,y) in self.deadEndPoses:
      fakeLay.layoutText[maxY - y][x] = 'x'"""
    #print(newlay)
    
    self.findDeadEnd(gameState)
    
    #print("===Find dead end poses=== 145\n", "\n",self.deadEndPoses)
    #print(gameState.data.layout)
    fakeLay = gameState.data.layout.deepCopy()
    #print("FAKE\n",fakeLay)
    maxY = self.height - 1
    newlay = ""
    for y in range(self.height):
      for x in range(self.width):
        if (x, maxY - y) in self.deadEndPoses:
          newlay +='X'
        else:
          newlay += fakeLay.layoutText[y][x]
        

        #layoutChar = fakeLay.layoutText[maxY - y][x]
        #print(layoutChar)
      newlay += '\n'
    """for (x,y) in self.deadEndPoses:
      fakeLay.layoutText[maxY - y][x] = 'x'"""
    #print(newlay)
    for enemy in self.enemies:
      belief[enemy] = util.Counter()
      self.trackPosition[enemy] = []
      # set our initial belief of the enemies
      belief[enemy][gameState.getInitialAgentPosition(enemy)] = 1
      self.trackPosition[enemy] += [gameState.getInitialAgentPosition(enemy)]
    self.enemyProbPos = util.Counter()
  def randomWalk(self, enemy, gameState):
    """
    Update our belief inference of enemy position when they CANNOT BE DETECTED.
    Generate a distribution based on random walk assumption of the enemies.
    In each step, all possible transitions are considered.
    """
    newBelief = util.Counter()
    global belief
    for pos in belief[enemy]:
      newPos = util.Counter()
      # get possible transition position
      transitPos =           [(pos[0], pos[1] + 1), 
        (pos[0] - 1, pos[1]), (pos[0], pos[1] ), (pos[0] + 1, pos[1]),    # disregard the option of STOP  
                              (pos[0], pos[1] - 1)]

      for tPos in transitPos:
        if tPos in self.legalPositions:
          if (tPos not in self.trackPosition[enemy]):    

            newPos[tPos] += 2    # more overlap more chances of ending up in this cell
            self.trackPosition[enemy] += [tPos]

          else:
            newPos[tPos] += 0.0001    # assume little possibility of going back
      newPos.normalize()
      
      for nPos, prob in newPos.items():
        # Update the probabilities for each of the positions.
        newBelief[nPos] += prob * belief[enemy][pos]    # transition probability = Pr(oldPos)*Pr(newPos)
  
    newBelief.normalize()
    belief[enemy] = newBelief  

  def observedEnemy(self, enemy, gameState):
    """
    This function updates our belief based on noisy observation and random walk result
    """
    noisyDist = gameState.getAgentDistances()[enemy]
    myPos = gameState.getAgentPosition(self.index)
    defendFoodCurrent = self.getFoodYouAreDefending(gameState).asList()
    
    newBelief = util.Counter()

    prevState = self.getPreviousObservation()
    if prevState:
      defendFoodPrev = self.getFoodYouAreDefending(prevState).asList()
      foodEaten = list(set(defendFoodPrev) - set(defendFoodCurrent))
      enemyState = prevState.getAgentState(enemy)
      invader = enemyState.isPacman

    global belief

    for legalPos in self.legalPositions:
      trueDist = self.getMazeDistance(legalPos, myPos)
      manhattanDist = util.manhattanDistance(myPos, legalPos)
      # given true distance, probability of noisy distance being true
      distProb = gameState.getDistanceProb(trueDist, noisyDist)
      
      # === eliminate unlikely readings ===
      if manhattanDist <= 5:
        newBelief[legalPos] = 0
      
      elif abs(trueDist - noisyDist) > 6:
        newBelief[legalPos] = 0

      else:
        newBelief[legalPos] += belief[enemy][legalPos] * distProb

      # === adding likely position ===
      if prevState:
        if foodEaten:    # if there's eaten food, enemy should be around that position now
          pos = foodEaten[0]
          if invader:
            transitPos =     [(pos[0], pos[1] + 1), 
          (pos[0] - 1, pos[1]), (pos[0], pos[1] ), (pos[0] + 1, pos[1]),    # disregard the option of STOP  
                                (pos[0], pos[1] - 1)]
            for tPos in transitPos:
              if tPos in self.legalPositions:
                newBelief[tPos] += 1
                self.trackPosition[enemy] += [tPos]

    if newBelief.totalCount() != 0:
      newBelief.normalize()
      belief[enemy] = newBelief

        
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
    timeLeft = gameState.data.timeleft/4    # for debug
    global belief
    
    # Update our belief of enemies position
    for enemy in self.enemies:
      enemyPos = gameState.getAgentPosition(enemy)
      if enemyPos:
        newBelief = util.Counter()
        newBelief[enemyPos] = 1
        belief[enemy] = newBelief
        
      else:
        self.randomWalk(enemy, gameState)
        self.observedEnemy(enemy, gameState)
        prevState = self.getPreviousObservation()
        if prevState:
          prevEnemyPos = prevState.getAgentPosition(enemy)
          prevMyPos = prevState.getAgentPosition(self.index)
          
          if prevEnemyPos and (self.getMazeDistance(prevEnemyPos, prevMyPos) == 1) and (prevState.getAgentState(enemy).isPacman):
            newBelief = util.Counter()
            newBelief[gameState.getInitialAgentPosition(enemy)] = 1
            print('enemy pacman busted')
            belief[enemy] = newBelief
            self.trackPosition[enemy] = []

    # Get most probable position of enemy

    for enemy in self.enemies:
      maxProb = sorted(belief[enemy].values())[-3:]    # choose top three probably positions
      probablePosition = [(pos, prob) for pos, prob in belief[enemy].items() if prob in maxProb]
      self.enemyProbPos[enemy] = probablePosition
    #print('\n=== 310 ===', enemyProbPos)


    # === ABOUT-TO-LOSE SENARIO ===
    # ACTION SENARIO 7: the enemy eat the Capsule
    scared = gameState.data.agentStates[self.index].scaredTimer
    enemyPacmanPos = self.checkStateSafeAtHome(gameState)
    if scared > 0 and enemyPacmanPos and not Pacman:
      #print('308 scared, the enermy eat the cap', scared, self.index, enemyPacmanPos)
      enermyIndex = [tup[0] for tup in enemyPacmanPos]
      toAct = self.minimax(gameState, self.index, enermyIndex, False)
      #print('=== 1140 ===scared, the enermy eat the cap', toAct,self.index,timeLeft,myPos)
      return toAct 

    if len(defendFood) <= self.totalFoodNum/5:
      
      values = [self.evaluatePatrol(gameState, a) for a in actions]
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      toAct = random.choice(bestActions)
      #print('=== 228 ===', self.index, toAct, timeLeft, myPos)
      return toAct

    # === NORMAL SENARIO ===
    # ACTION STEP 1: reach frontier first
    if self.actionFrontier:
      toAct = self.actionFrontier.pop(0)
      #('=== 351 ===', self.index, toAct, timeLeft)
      return toAct

    # ACTION STEP 2: after reaching frontier
    # === REACHED-AN-IMPASSE SENARIO ===
    isInImpasse = self.reachedImpasse(gameState, myPos)
    if isInImpasse:
      values = [self.evaluateImpasse(gameState, a) for a in actions]
      
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      bestAction = random.choice(bestActions)
      #print('=== 364 === impasse', self.index, bestAction, timeLeft,myPos)
      #print('\n=== 364 ===', enemyProbPos)

      return bestAction
    
    # === DETECTS GHOST === USE MINIMAX ===
    if gstPos and Pacman:
      enermyIndex = [tup[0] for tup in gstPos]
      #depth = self.miniMaxDepth
      
      toAct = self.minimax(gameState, self.index, enermyIndex) 
      #print("minimax 316", self.index, toAct, myPos)    
      return toAct

    # ACTION SENARIO 6: about to win, after detect ghost
    foodLeft = len(self.getFood(gameState).asList())
    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      #print('=== 333 ===', self.index, bestAction, timeLeft, myPos)
      return bestAction

    # ACTION SENARIO 4: if food more than threshold, need to cash in
    goHome = self.needToCashIn(myPos, myState, self.minPelletsToCashIn, timeLeft)
    notHome = self.distToHome[myPos]
    isCloseToFood = self.isCloseToFood(gameState, actions)
    if goHome and (notHome) and (not isCloseToFood):
      dist, end = self.distToHome[myPos]
      # BFS find shortest path to home
      _, path = self.getBfsPath(gameState, end, myPos)
      toAct = path.pop(0)

      # check if threatened by ghost
      threatToHome = self.checkStateSafe(gameState)
      Pacman = gameState.getAgentState(self.index).isPacman

      # escape
      if threatToHome and Pacman:
        enermyIndex = [tup[0] for tup in gstPos]
        
        toAct = self.minimax(gameState, self.index, enermyIndex) 
       # print('=== 406 === need to cash in and threatened action', toAct)    
        return toAct
      #print('=== 408 === need to cash in and no threat action', toAct)
      return toAct

    
    

    # ACTION SENARIO 5: explore food or capsule
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    toAct = random.choice(bestActions)
    #print('=== 366 ===', self.index, toAct, timeLeft)
    return toAct


  # helper function for findDeadEnd, to detect the number of wall (includes filled cell) surrend to one position 
  def detectSurroundWallWithFilled(self, gameState, pos, filledPos):
    wallNum = 0
    if gameState.hasWall(pos[0] + 1, pos[1]) or (pos[0] + 1, pos[1]) in filledPos:
      wallNum += 1
    if gameState.hasWall(pos[0] - 1, pos[1]) or (pos[0] - 1, pos[1]) in filledPos:
      wallNum += 1
    if gameState.hasWall(pos[0], pos[1] + 1) or (pos[0], pos[1] + 1) in filledPos:
      wallNum += 1
    if gameState.hasWall(pos[0], pos[1] - 1) or (pos[0], pos[1] - 1) in filledPos:
      wallNum += 1
    return wallNum

  # Find all dead end positions in the maze
  def findDeadEnd(self, gameState):
    deadEndPos = set()
    posQueue = util.Queue()
    for x in range(1, self.width-1):
      #print("x", x, self.height-1)
      for y in range(1, self.height-1):
        if not gameState.hasWall(x,y) and self.detectSurroundWall(gameState, (x,y)) == 3:
          
          deadEndPos.add((x,y))
          posQueue.push((x,y))
    while not posQueue.isEmpty():
      curPos = posQueue.pop()
      #deadEndPos.add(curPos)
      neighborPoses = [(curPos[0]+1, curPos[1]),(curPos[0]-1, curPos[1]),(curPos[0], curPos[1]+1),(curPos[0], curPos[1]-1)]
      #neighborPoses = [(curPos[0]+i, curPos[1]+j) for i in [-1, 0, 1] for j in [-1, 0, 1] if not gameState.hasWall(curPos[0]+i, curPos[1]+j)]
      for neighborPos in neighborPoses:

        if neighborPos not in deadEndPos and not gameState.hasWall(neighborPos[0],neighborPos[1]):
          if self.detectSurroundWallWithFilled(gameState, neighborPos, deadEndPos) >= 3:
            deadEndPos.add(neighborPos)
            posQueue.push(neighborPos)
    self.deadEndPoses = deadEndPos



      


       
        

    

   
    

      


    





   
  
  def checkStateSafeAtHome(self, gameState):
    enemy = self.getEnemy(gameState)
    enemyPacman = enemy['Pacman']
    agentPos = gameState.getAgentPosition(self.index)
    nearbyEnermy = []

    for index, pos in enemyPacman:
      dist = self.getMazeDistance(agentPos, pos)
      if dist <= 5:
        nearbyEnermy.append((index, pos))

    if not nearbyEnermy:
      return None
    return nearbyEnermy

  
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

    # is Pacman
    features['isPacman'] = successor.getAgentState(self.index).isPacman

    return features

  def getWeightsGoHome(self,gameState, action):
    return {'homeDist': 50, 'distToGhost': 10, 'isPacman': -80}

  def reachedImpasse(self, gameState, myPos):
    inImpasseRegion = bool(myPos in self.frontierPoints)
    #print('=== 320 ===', friendPos, myPos, gstPos)
    gstPos = self.checkStateSafe(gameState)
    return inImpasseRegion and gstPos

  def evaluateImpasse(self, gameState, action):
    features = self.getFeaturesImpasse(gameState, action)
    weights = self.getWeightsImpasse(gameState, action)
    #print(action, features, features*weights)
    return features * weights

  def getFeaturesImpasse(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    nextPos = successor.getAgentState(self.index).getPosition()


    # Compute distance to the nearest ghost free (i.e. least belief in ghost position) enemy territory
    features['distToGstFreeEnemyAreaY'] = self.getDistToGstFreeEnemyAreaY(gameState, nextPos)
    features['distToGstFreeEnemyAreaX'] = self.getDistToGstFreeEnemyAreaX(gameState, nextPos)

    # Away from ghost
    ghsPosition = self.checkStateSafe(gameState)
    if ghsPosition:
      for enemyIdx, pos in ghsPosition:
        features['distToGhost'] += self.getMazeDistance(pos, nextPos)
    
    features['distToGhost'] += self.getApproxGhostDistance(nextPos)
    
    # to avoid the situation that the destination of the action has a ghost, nextpos will be starting point.
    if nextPos == self.start:
      features['distToGhost'] = 0
    
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
      if self.mayBeEaten(nextPos, ghsPosition) or nextPos == self.start:
        features['isEaten'] = 1
      else:
        features['isEaten'] = 0
    return features

  def getDistToGstFreeEnemyAreaY(self, gameState, nextPos):
    """
    Maximise the abs of y axis distance to the belief of ghost position
    """
    dist = 0
    global belief
    posList = list(belief[self.enemies[0]].items()) + list(belief[self.enemies[1]].items())
    for pos, prob in posList:
      if pos in self.enemyCells:

        dist += prob*(abs(pos[1] - nextPos[1]))    # maximise y distance
    return -1*dist
  def getDistToGstFreeEnemyAreaX(self, gameState, nextPos):
    """
    Maximise the abs of y axis distance to the belief of ghost position
    """
    dist = 0
    global belief
    posList = list(belief[self.enemies[0]].items()) + list(belief[self.enemies[1]].items())
    for pos, prob in posList:
      if pos in self.enemyCells:

        dist += prob*(abs(pos[0] - nextPos[0]))    # maximise y distance
    return -1*dist


  def minimax(self, gameState, playerIndex, enermyIndex, isPacman = True):
    if len(enermyIndex) == 1:
      allIndexes = [self.index, enermyIndex[0]]
      depth = self.miniMaxDepth
      #print("+++ 1 Enermy MINIMAX BEGIN+++551")
      _, toAct = self.max2(gameState, depth, self.index, allIndexes, isPacman)
      #print("+++1 Enermy MINIMAX RESULT 553", toAct)
    elif len(enermyIndex) == 2:
      #actions = gameState.getLegalActions(self.index)
      allIndexes  = [playerIndex] + enermyIndex
      depth = self.miniMaxDepth2
      #print("+++ 2 Enermy MINIMAX BEGIN+++558")
      _, toAct = self.maxn(gameState, depth, playerIndex, allIndexes, isPacman)
      #toAct = random.choice(actions)
      #print("HAVNT IMPLEMENTED")
      #print("+++2 Enermy MINIMAX RESULT 562", toAct)
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
    

  def getApproxGhostDistance(self, nextPos):

    dist = 0
    global belief
    posList = list(belief[self.enemies[0]].items()) + list(belief[self.enemies[1]].items())
    for pos in posList:

      if pos in self.enemyCells:
        dist += belief*self.getMazeDistance(pos, nextPos)
    return dist

  def updateBelief(self, position, idx):
    alreadyExistedPositions = belief[idx].keys()
    #print('===408 ===', alreadyExistedPositions)
    if position in alreadyExistedPositions or (not alreadyExistedPositions):
      possiblePositions = [(position[0]+i, position[1]+j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (position[0]+i, position[1]+j) in self.legalPositions]
      for pos in possiblePositions:
        belief[idx][pos] += 1/9
        #print('===413 ===', alreadyExistedPositions)
      belief[idx][position] += 1/9
    else:
      belief[idx] = util.Counter()    # if out of threat, clean our belief

  def getDistToFriend(self, friendPos, myPos):
    favoredY = self.favoredY
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
    return {'distToGstFreeEnemyAreaY': -30, 'distToGstFreeEnemyAreaX': -10, 'distToGhost': 40,\
    'stop': -12, 'reverse': -5, 'invaderDistance': -6, \
    'isPacman': -3, 'isEaten': -80}


  def max2(self, gameState, depth, playerIndex, allGameIndexes, isPacman):
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
          actionValue,a = self.min2(successor, depth-1,enermyIndex, allGameIndexes, isPacman)
          myPos = successor.getAgentPosition(playerIndex)
          myPosList.append(myPos)
          #print("352++++max ",depth-1, actionValue, action, a)
          actionValues.append(actionValue)
          applicableActions.append(action)
      
      # when the final action (depth = self.minimaxDepth) list is empty, return 'STOP'.
      if len(applicableActions) == 0:
        return 0, 'Stop'

      maxValue = max(actionValues)
      bestActions = [a for a, v, p in zip(applicableActions, actionValues,myPosList) if v == maxValue]
      bestActionsPos = [p for a, v, p in zip(applicableActions, actionValues,myPosList) if v == maxValue]

      if depth!= self.miniMaxDepth:
        return maxValue, random.choice(bestActions)

      # select next action based on features (try to avoid go to the dead end)
      if depth == self.miniMaxDepth:
        #print("664 MINIMAX best actions",bestActions,maxValue)
        selectedActions = self.selectMiniMaxAction(bestActions, bestActionsPos, gameState, isPacman)
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
    if gameState.hasWall(pos[0], pos[1] - 1):
      wallNum += 1
    return wallNum
    

          
  # Select actions with shortest distance to home from the highest depth   
  def selectMiniMaxAction(self, bestActions, bestActionsPos, gameState, isPacman):
    if isPacman:
      selectedActionsAtHome = []
      distToHomeList = []
      selectedActionsInEnermy = []
      actionsInEnermy = []
      actionsInDeadEnd = []
      actionsInDeadEndDist = []
      for bestAction, pos in zip(bestActions, bestActionsPos):
        if pos not in self.enemyCells:
          selectedActionsAtHome.append(bestAction)
        elif pos in self.enemyCells and pos not in self.deadEndPoses:
          distToHomeList.append(self.distToHome[pos][0])
          actionsInEnermy.append(bestAction)
        elif pos in self.enemyCells:
          actionsInDeadEnd.append(bestAction)
          actionsInDeadEndDist.append(self.distToHome[pos][0])

  
      if len(distToHomeList) !=0:
        closestToHomeDist = min(distToHomeList)
        selectedActionsInEnermy = [a for a,d in zip(actionsInEnermy, distToHomeList) if d == closestToHomeDist]

        selectedActions = selectedActionsAtHome + selectedActionsInEnermy
      elif len(selectedActionsAtHome):
        selectedActions = selectedActionsAtHome 
      else:
        closestToHomeDistInDeadEnd = min(actionsInDeadEndDist)
        selectedActionsInEnermyDeadEnd = [a for a,d in zip(actionsInDeadEnd, actionsInDeadEndDist) if d == closestToHomeDistInDeadEnd]
        selectedActions = selectedActionsInEnermyDeadEnd
      if len(selectedActions) != 0:
        #print('407--', random.choice(selectedActions))
        return selectedActions
        #print('408--', random.choice(selectedActions))
      else:
        return bestActions
    else:
      selectedActionsAtHome = []
      distToEnemyHomeList = []
      selectedActionsInEnermy = []
      actionsAtHome = []
      actionsInDeadEnd = []
      actionsInDeadEndDist = []
      for bestAction, pos in zip(bestActions, bestActionsPos):
        if pos in self.enemyCells:
          selectedActionsInEnermy.append(bestAction)
        elif pos not in self.enemyCells and pos not in self.deadEndPoses:
          distToEnemyHomeList.append(self.computeDistToEnemyHome(pos)[0])
          actionsAtHome.append(bestAction)
        elif pos not in self.enemyCells:
          actionsInDeadEnd.append(bestAction)
          actionsInDeadEndDist.append(self.computeDistToEnemyHome(pos)[0])

  
      if len(distToEnemyHomeList) !=0:
        closestToEnemyHomeDist = min(distToEnemyHomeList)
        selectedActionsAtHome = [a for a,d in zip(actionsAtHome, distToEnemyHomeList) if d == closestToEnemyHomeDist]

        selectedActions = selectedActionsAtHome + selectedActionsInEnermy
      elif len(selectedActionsInEnermy):
        selectedActions = selectedActionsInEnermy
      else:
        closestToHomeDistInDeadEnd = min(actionsInDeadEndDist)
        selectedActionsInEnermyDeadEnd = [a for a,d in zip(actionsInDeadEnd, actionsInDeadEndDist) if d == closestToHomeDistInDeadEnd]
        selectedActions = selectedActionsInEnermyDeadEnd

      if len(selectedActions) != 0:
        #print('407--', random.choice(selectedActions))
        return selectedActions
        #print('408--', random.choice(selectedActions))
      else:
        return bestActions  
  
  def computeDistToEnemyHome(self,pos):
    minDist = 9999
    for loc in self.frontierPointsEnemy:
      curDist = self.getMazeDistance(loc, pos)
      if curDist < minDist:
        minDist = curDist
        minDistLoc = loc
    return curDist, minDistLoc

  def min2(self, gameState, depth, playerIndex, allIndexes, isPacman):
    bestActionValue = 9999
    bestAction = None
    actions = gameState.getLegalActions(playerIndex)
    #value = util.Counter()
    for action in actions:
      if action!= 'Stop':
        successor = gameState.generateSuccessor(playerIndex, action)
        nextIndex = allIndexes[0]
        actionValue,a = self.max2(successor, depth-1, nextIndex, allIndexes, isPacman)
        #print("369++++",depth-1, actionValue, action,a)
        if bestActionValue > actionValue:
          bestAction = action
          bestActionValue = actionValue
    return bestActionValue, bestAction
  
  
  def maxn(self, gameState, depth, playerIndex, allIndexes, isPacman):
    if depth == 0 or gameState.getLegalActions(playerIndex) == None or gameState.isOver():
      #print(depth, gameState)
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
          actionValue,a = self.maxn(successor, depth-1,enermyIndex, allIndexes, isPacman)
          myPos = successor.getAgentPosition(playerIndex)
          for i in range(len(actionValue)):
            actionValues.append(actionValue[i])
            applicableActions.append(action)          
            myPosList.append(myPos)
                             
      # when the final action (depth = self.minimaxDepth) list is empty, return 'STOP'.
      if len(applicableActions) == 0:
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
        #print("831", depth,toAct,bestActionValueTuples)
        return bestActionValueTuples, toAct

      # select next action based on features (try to avoid go to the dead end)
      if depth == self.miniMaxDepth2:
        #closestToHome = []
        #print("minimax n 812", bestActions,bestActionValueTuples)
        selectedActions = self.selectMiniMaxAction(bestActions, bestActionsPos, gameState, isPacman)
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

    global belief

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

    # distance to belief of where the enemies are
    distBelief = 0
    posList = self.enemyProbPos[self.enemies[0]] + self.enemyProbPos[self.enemies[1]]
    if posList:
      for pos, prob in posList:
        if pos in self.myCells:
          distBelief += self.getMazeDistance(myPos, pos)*prob
      features['distToBelief'] = distBelief
      #print('\n====\n patrol', posList)
    else:
      features['distToBelief'] = 0

    return features
  
  def getDistToPatrol(self, myPos, patrolArea):
    dists = 0
    i = 0
    for pos in patrolArea:
      dists += (self.getMazeDistance(pos, myPos))
      i += 1
    return dists/i

  def getWeightsPatrol(self, gameState, action):
    return {'numInvaders': -70, 'onDefense': 100, 'distToPatrol': -10, 'invaderDistance': -20, 'distToBelief': -12}

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

  def needToCashIn(self, myPos, nextState, maxCarry, timeLeft):
    # if we have enough pellets, attempt to cash in
    distToHome = self.distToHome[myPos]
    if distToHome:
      if nextState.numCarrying >= maxCarry or timeLeft < 2*distToHome[0]:
        return 1
      else:
        return 0
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
    nearbyGhost = []
    # check if ghost scared
    for index, pos in enemyGhost:
      dist = self.getMazeDistance(agentPos, pos)
      scared = gameState.data.agentStates[index].scaredTimer
      if dist <= 5 and scared <=2 :
        nearbyGhost.append((index, pos))

    if not nearbyGhost:
      return None
    return nearbyGhost
    
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
    foodLeft = len(self.getFood(gameState).asList())
    timeLeft = gameState.data.timeleft/4

    # Update our belief of enemies position
    for enemy in self.enemies:
      enemyPos = gameState.getAgentPosition(enemy)
      if enemyPos:
        newBelief = util.Counter()
        newBelief[enemyPos] = 1
        belief[enemy] = newBelief
        
      else:
        #self.randomWalk(enemy, gameState)
        #self.observedEnemy(enemy, gameState)
        prevState = self.getPreviousObservation()
        if prevState:
          prevEnemyPos = prevState.getAgentPosition(enemy)
          prevMyPos = prevState.getAgentPosition(self.index)
          
          if prevEnemyPos and (self.getMazeDistance(prevEnemyPos, prevMyPos) == 1) and (prevState.getAgentState(enemy).isPacman):
            newBelief = util.Counter()
            newBelief[gameState.getInitialAgentPosition(enemy)] = 1
            #print('enemy pacman busted')
            belief[enemy] = newBelief
            self.trackPosition[enemy] = []
    #print(' === 303 === updated belief of random walk', self.belief)
    # Get most probable position of enemy
    
    for enemy in self.enemies:
      maxProb = sorted(belief[enemy].values())[-1:]
      probablePosition = [(pos, prob) for pos, prob in belief[enemy].items() if prob in maxProb]
      self.enemyProbPos[enemy] = probablePosition
    #print('\n=== 310 ===', enemyProbPos)


    
    # ACTION SENARIO : the enemy eat the Capsule,run away, tend to move to the enermy's side
    scared = gameState.data.agentStates[self.index].scaredTimer
    enemyPacmanPos = self.checkStateSafeAtHome(gameState)
    #print('308 scared', scared, enemyPacmanPos)
    if scared > 0 and enemyPacmanPos and not Pacman:
      enermyIndex = [tup[0] for tup in enemyPacmanPos]
      toAct = self.minimax(gameState, self.index, enermyIndex, False)
      #print('=== 1103 enermy eat cap ===', self.index, toAct, timeLeft, myPos)
      return toAct

    # ACTION SENARIO : when pacman and in enermy's side threatened by the ghost
    gstPos = self.checkStateSafe(gameState)
    
    if gstPos and Pacman:
      enermyIndex = [tup[0] for tup in gstPos]
      #depth = self.miniMaxDepth
      
      toAct = self.minimax(gameState,self.index, enermyIndex)   
      #print('=== 1114 minimax ===', self.index, toAct, timeLeft,myPos)  
      return toAct
    
    # === ABOUT TO LOSE SENARIO ===
    if len(defendFood) <= self.totalFoodNum/5:
      values = [self.evaluatePatrol(gameState, a) for a in actions]
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      bestAction = random.choice(bestActions)
     # print('=== 1123 ===', self.index, bestAction, timeLeft,myPos)
      return bestAction

     

    
    # === OFFENSIVE SENARIO ===
    if len(defendFood) > self.totalFoodNum/2 and timeLeft > 90:

      # CASE 1: REACHED-AN-IMPASSE SENARIO ===
      isInImpasse = self.reachedImpasse(gameState, myPos)
      if isInImpasse:
        
        values = [self.evaluateImpasse(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        bestAction = random.choice(bestActions)
        #print('=== 1165 === impasse', self.index, bestAction, timeLeft,myPos)
        #print('\n=== 1166 ===', enemyProbPos)
        return bestAction
      

      # === ABOUT-TO-WIN SENARIO ===
      if gstPos and Pacman:
        enermyIndex = [tup[0] for tup in gstPos]
        #depth = self.miniMaxDepth
        toAct = self.minimax(gameState, self.index, enermyIndex) 
        #print('=== 1149 ===', self.index, toAct, timeLeft,myPos)    
        return toAct

      if foodLeft <= 2:  
        bestDist = 9999
        for action in actions:
          successor = self.getSuccessor(gameState, action)
          pos2 = successor.getAgentPosition(self.index)
          dist = self.getMazeDistance(self.start,pos2)
          if dist < bestDist:
            bestAction = action
            bestDist = dist
        #print('=== 1161 ===', self.index, bestAction, timeLeft,myPos) 
        return bestAction

      # CASE 2: carrying enough food, go home
      goHome = self.needToCashIn(myPos, myState, self.minPelletsToCashIn, timeLeft)
      notHome = self.distToHome[myPos]
      isCloseToFood = self.isCloseToFood(gameState, actions)
      if goHome and (notHome) and (not isCloseToFood):
        dist, end = self.distToHome[myPos]
        # BFS find shortest path to home
        _, path = self.getBfsPath(gameState, end, myPos)
        toAct = path.pop(0)

        # check if threatened by ghost
        threatToHome = self.checkStateSafe(gameState)
        Pacman = gameState.getAgentState(self.index).isPacman

        # escape
        if threatToHome and Pacman:
          enermyIndex = [tup[0] for tup in gstPos]
          
          toAct = self.minimax(gameState, self.index, enermyIndex) 
          #print('=== 1204 === need to cash in and threatened action', toAct)    
          return toAct
        #print('=== 1205 === need to cash in and no threat action', toAct)
        return toAct
      

      # CASE 3: no threats and still hungry  
      values = [self.evaluate(gameState, a) for a in actions]

    # === DEFENDIVE SENARIO ===
    else:
      values = [self.evaluateDefensive(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    
    
    bestAction = random.choice(bestActions)
    #print('=== 1200 ===', self.index, bestAction, timeLeft,myPos)
    return bestAction

  

  def getWeightsImpasse(self, gameState, action):
    # Give more incentive to intercept enemy pacman
    return {'distToGstFreeEnemyAreaY': -25, 'distToGstFreeEnemyAreaX': -10, 'distToGhost': 32,\
    'stop': -12, 'reverse': -5, 'invaderDistance': -4, \
    'isPacman': -3, 'isEaten': -80}

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
