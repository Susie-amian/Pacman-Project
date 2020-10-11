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
    
    self.start = gameState.getAgentPosition(self.index)
    self.minPelletsToCashIn = 5

    # get layout frontier
    self.width = gameState.data.layout.width
    self.height = gameState.data.layout.height
    #print([(i, j) for i in range(self.width) for j in range(self.height)])

    self.midPointLeft = int((self.width / 2.0)-1)
    self.midPointRight = int((self.width / 2.0)+1)
    self.isRed = gameState.isOnRedTeam(self.index)

    if self.isRed:
      self.midPoint = int(self.midPointLeft)
      self.midPointEnemy = int(self.midPointRight)
      self.enemyCells = []
      for i in range(self.midPointRight-1, self.width):
        for j in range(self.height):
          if not gameState.hasWall(i, j):
            
            self.enemyCells.append((i, j))
      
    else:
      self.midPoint = int(self.midPointRight)
      self.midPointEnemy = int(self.midPointLeft)
      self.enemyCells = []
      for i in range(self.midPointLeft):
        for j in range(self.height):

          if not gameState.hasWall(i, j):
            
            self.enemyCells.append((int(i), int(j)))

    self.frontierPoints = [(self.midPoint, int(i)) for i in range(self.height) if not gameState.hasWall(self.midPoint, i)]
    self.frontierPointsEnemy = [(self.midPointEnemy, int(i)) for i in range(self.height) if not gameState.hasWall(self.midPointEnemy, i)]

    self.distToHome = self.getDistToHome(self.frontierPoints, self.enemyCells)
    # get shortest path to frontier point
    
    self.frontierState, self.actionFrontier = self.toFrontier(gameState, self.frontierPoints, self.start)

    # get shortest path from frontier to power capsule
    self.capsulePosition = self.getCapsules(gameState)[0]
    self.closestFrontier = self.frontierState.getAgentPosition(self.index)
    self.capsulState, self.actionCapsule = self.getBfsPath(self.frontierState, self.capsulePosition, self.closestFrontier)
    
    # minimax initial set up
    self.miniMaxDepth = 3
    print(self.frontierPointsEnemy)


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
    Returens the closest path to home frontier
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
    # reach frontier first
    if self.actionFrontier:
      toAct = self.actionFrontier.pop(0)
      return toAct

    # after reaching frontier
    if self.actionCapsule:
      toAct = self.actionCapsule.pop(0)

      # check if planned step is safe
      gstPos = self.checkStateSafe(gameState)
      if not gstPos:    # safe, ghost not detected
        return toAct

      # otherwise, ghost detected, run.
      self.actionCapsule = []    # ditch all pre-planed path

    # if detects ghost, use minimax to get out of the way
    gstPos = self.checkStateSafe(gameState)
    myPos = gameState.getAgentPosition(self.index)
    Pacman = gameState.getAgentState(self.index).isPacman

    if gstPos and Pacman:
      enemyPos = [tup[1] for tup in gstPos]
      toAct = self.getMiniMaxAction(gameState, myPos, enemyPos)
      return toAct

    # if food more than threshold, need to cash in
    myState = gameState.getAgentState(self.index)
    myPos = gameState.getAgentPosition(self.index)

    goHome = self.needToCashIn(myPos, myState, 5)
    isHome = self.distToHome[myPos]
    if goHome and (not isHome):
      dist, end = self.distToHome[myPos]
      state, path = self.getBfsPath(gameState, end, myPos)
      toAct = path.pop(0)
      return toAct

    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

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
      return bestAction

    return random.choice(bestActions)


  def getEvaluation(self, myPos, enemyPos):
    #print('=== 239 === ', myPos)
    total = self.getMazeDistance(myPos, enemyPos)
    #if self.distToHome[myPos]: 
    #  total += (-1*self.distToHome[myPos][0])/2
    return total

  def getMiniMaxAction(self, gameState, myPos, enemyPos):
    toAct = self.miniMax(gameState, self.miniMaxDepth, self.index, True, myPos = myPos, enemyPos = enemyPos)[1]
    
    return toAct
    
  def miniMax(self, gameState, depth, agent, maximizing, myPos, enemyPos):
    """
    Returns an action by miniMax algorithm
    """

    ghosts = self.checkStateSafe(gameState)
    #myPos2 = gameState.getAgentPosition(agent)
    
    if depth == 0 or (not ghosts):
      #print('=== 255 DEPTH REACHED === RETURN VALUE:', self.getEvaluation(myPos, enemyPos))
      #print('=== 261 THE DIFFERENCE BETWEEN POINTS ===:', myPos)
      return self.getEvaluation(myPos, enemyPos), Directions.STOP

    minDist = 9999
    ghostIndex = 0
    enemyPos = (0, 0)
    ghostPos = [tup[1] for tup in ghosts]

    for idx, pos in ghosts:
      threatDist = self.getMazeDistance(pos, myPos)
      if minDist > threatDist:
        minDist = threatDist
        ghostIndex = idx
        enemyPos = pos

    if maximizing:
      #print('=== 272 MAXIMIZING ===', myPos, enemyPos)
      return self.maximizer(gameState, depth, agent = self.index, enemy = ghostIndex, myPos = myPos, enemyPos = enemyPos)
    else:
      #print('=== 275 MiniMIZING ===', myPos, enemyPos)
      return self.minimizer(gameState, depth, agent = ghostIndex, enemy = self.index, myPos = enemyPos, enemyPos = myPos)

  
  def minimizer(self, game_state, depth, agent, enemy, myPos, enemyPos):
    #print('\n=== 280 INSIDE MINIMIZER ===')
    actions = game_state.getLegalActions(agent)
    scores = []

    for action in actions:
      #print('285 CURRENT ACTION CHOSEN BY MINIMIZER', action)
      successor_game_state = game_state.generateSuccessor(agent, action)
      myPos = successor_game_state.getAgentPosition(agent)
      #print('294 MY POSITION', myPos)
      scores.append(self.miniMax(successor_game_state, depth - 1, agent = self.index, maximizing=True, myPos = myPos, enemyPos = enemyPos)[0])
      
    min_score = min(scores)
    min_indexes = [i for i, score in enumerate(scores) if score == min_score]
    chosen_action = actions[random.choice(min_indexes)]

    return min_score, chosen_action

  def maximizer(self, gameState, depth, agent, enemy, myPos, enemyPos):
    #print('\n=== 298 INSIDE MAXIMIZER ===')
    actions = gameState.getLegalActions(agent)
    scores = []

    for action in actions:
      #print('303 CURRENT ACTION CHOSEN BY MAXIMIZER', action)
      successorGameState = gameState.generateSuccessor(agent, action)
      myPos = successorGameState.getAgentPosition(agent)
      #print('311 MY POSITION', myPos)
      scores.append(self.miniMax(successorGameState, depth, agent = enemy, maximizing = False, myPos = myPos, enemyPos = enemyPos)[0])

    max_score = max(scores)
    max_indexes = [i for i, score in enumerate(scores) if score == max_score]
    chosen_action = actions[random.choice(max_indexes)]

    return max_score, chosen_action

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
      minDistance = min([self.getMazeDistance(nextPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # Distance to Power Capsule
    capsule = self.getCapsules(gameState)
    if capsule:
      capsuleDist = self.getMazeDistance(capsule[0], nextPos)
      features['distanceToCapsule'] = capsuleDist
    else:
      features['distanceToCapsule'] = 0 # since eaten

    # Away fron ghost
    ghsPosition = self.checkStateSafe(gameState)
    if ghsPosition:
      for idx, pos in ghsPosition:
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


    features['numInvaders'] = len(invaders)

    # isPacman
    features['isPacman'] = successor.getAgentState(self.index).isPacman

    return features

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
    'numInvaders': -2, 'isPacman': 3}


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
      #print('DISTANCE', dist)
      if dist < minDist:
        minDist = dist
    if minDist > 5:
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


class DefensiveReflexAgent(ClassicPlanAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)


    myState = gameState.getAgentState(self.index)
    myPos = gameState.getAgentPosition(self.index)
    timeLeft = gameState.data.timeleft
    defendFood = self.getFoodYouAreDefending(gameState).asList()

    if len(defendFood) > 12 and timeLeft > 80:

      goHome = self.needToCashIn(myPos, myState, 5)
      if goHome:
        dist, end = self.distToHome[myPos]

        print('=== end ===', myPos, end)
        state, path = self.getBfsPath(gameState, end, myPos)
        toAct = path.pop(0)
        return toAct

      threatToHome = self.checkStateSafe(gameState)
      isHome = (not self.distToHome[myPos])
      if threatToHome and (not isHome):

        dist, home = self.distToHome[myPos]

        _, actionSeq = self.getBfsPath(gameState, home, myPos)
        toAct = actionSeq.pop(0)
        actedPos = (gameState.generateSuccessor(self.index, toAct)).getAgentPosition(self.index)
        for idx, gstPos in threatToHome:
          if self.getMazeDistance(actedPos, gstPos) < 2:
            toAct = self.escape(actions, gameState)
        return toAct
        

      values = [self.evaluate(gameState, a) for a in actions]
    else:
      values = [self.evaluateDefensive(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

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
      return bestAction

    return random.choice(bestActions)


  def escape(actions, gameState):
    features = self.getFeaturesEscape(gameState, action)
    weights = self.getWeightsEscape(gameState, action)

    vals = features * weights
    maxValue = max(values)

    bestActions = [act for act, val in zip(actions, vals) if val == maxval]

    return random.choice(bestActions)

  def getFeaturesEscape(gameState, action):
    myPos = gameState.getAgentPosition(self.index)
    features['successorScore'] = -len(foodList)
    ghsPosition = self.checkStateSafe(gameState)
    if ghsPosition:
      for idx, pos in ghsPosition:
        features['distToGhost'] += self.getMazeDistance(pos, nextPos)
    else: 
      features['distToGhost'] = 0
    features['toHome'] = -self.distToHome[myPos]
    return features

  def getWeightsEscape(gameState, action):
    return {'successorScore': 2, 'distToGhost': 40, 'toHome': 10}


  def getWeights(self, gameState, action):
    return {'successorScore': 120, 'distanceToFood': -1, \
    'distanceToCapsule': -2, 'distToGhost': 30, 'cashIn': 0, \
    'stop': -15, 'reverse': -2, 'invaderDistance': -2, \
    'numInvaders': -3, 'isPacman': 3}

################################
# BELOW IS FOR DEFENSIVE STATE #
################################

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
        print('===', inv)
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


    return features

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
    return {'numInvaders': -70, 'onDefense': 100, 'invaderDistToHome': 30, 'invaderDistance': -20, 'stop': -100, 'reverse': -3}
