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

CHEAT = False
beliefs = []
beliefsInitialized = []
MINIMUM_PROBABILITY = .0001
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'QlearningAgent', second = 'RandomNumberTeamAgent'):
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

    '''
    Your initialization code goes here, if you need any.
    '''

    self.epsilon = 0.9
    self.alpha = 0.4
    self.discount = 0.9

    self.start = gameState.getAgentPosition(self.index)
    self.totalFoodList = self.getFood(gameState).asList()

    # recording dict
    self.weights = util.Counter()

    self.weights['successorScore'] = 1
    self.weights['distToFood'] = 1

    self.goalFood, self.DistGoalFood = self.getDistToFood(gameState)

  def chooseAction(self, gameState):
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

    # update the weight
    self.update(gameState, actionToReturn, successor)

    # return the action
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
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    # Score
    features['successorScore'] = self.getScore(successor)  # self.getScore(successor)

    # Distance to closest food
    _, minDist = self.getDistToFood(successor)
    features['distToFood'] = 1/minDist

    _, minDist = self.getDistToFood(successor,sucFoodList)
    #features['eatenFood'] = len(self.eatenFood)
    #features['distToFood'] = minDist
    features['distToFood'] = 1/minDist
    return features

  def getDistToFood(self, currentState):
    pos = currentState.getAgentState(self.index).getPosition()
    foodList =  self.getFood(currentState).asList()
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

  def getReward(self, gameState, toAct):
    # init reward
    reward = 0
    curState = gameState
    curPos = curState.getAgentPosition(self.index)
    curFoodNum = len(self.getFood(curState).asList())

    sucState = gameState.generateSuccessor(self.index, toAct)
    sucPos = sucState.getAgentState(self.index).getPosition()
    sucFoodNum = len(self.getFood(sucState).asList())

    # better score
    if sucState.getScore() > curState.getScore():
      reward += 20
    
    # food difference
    foodDifference = sucFoodNum - curFoodNum
    if (foodDifference > 0):
      reward += 1000
    elif (foodDifference < 0):
      reward += foodDifference*2

    # closer distance to capsule

    # closer distance to food
    if self.goalFood == sucPos:
      self.goalFood, sucDist = self.getDistToFood(sucState)
    else:
      sucDist = self.getMazeDistance(sucPos, self.goalFood)
   
    reward += self.getFoodProximityReward(curPos, sucDist, self.goalFood)
    return reward



  def getFoodProximityReward(self, prevPos, curDist, goalFood):
    prevDist = self.getMazeDistance(prevPos, goalFood)
    return 100*(prevDist-curDist) 

  def update(self, gameState, action, nextState):
    
    reward = self.getReward(gameState, action) #value
    
    oldQ = self.getQvals(gameState, action) #value
    newMaxQval= self.computeValueFromQValues(nextState) #value
    learnedWeight = self.alpha*(reward + self.discount * newMaxQval - oldQ)  #value 
    features = self.getFeatures(gameState, action)  #counter

    for key in features.keys():
      self.weights[key] += learnedWeight * features[key]


class RandomNumberTeamAgent(CaptureAgent):
    counter = None  # type: int

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

        # set variables
        self.count = 0
        self.lastCount = 0
        self.patrolPositions = self.getPatrolPosition(gameState)

        self.walls = gameState.getWalls()
        self.start = gameState.getAgentPosition(self.index)
        self.target = None
        actions = ['Stop', 'North', 'West', 'South', 'East']
        self.totalFoodList = self.getFood(gameState).asList()

        self.FoodLastRound = None

        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height

        self.initialDefendingPos = self.patrolPositions[int(len(self.patrolPositions)/2)]
        #print(self.initialDefendingPos)

        self.lastGoal1 = None
        self.lastGoal2 = None

        # action plans
        self.path = []
        self.lastState = None
        self.lastAction = None
        self.posNum = 0

        self.lastFoodList = self.getFood(gameState).asList()

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        if gameState.isOnRedTeam(self.index) and "East" in actions:
            actions.remove("East")
        elif not gameState.isOnRedTeam(self.index) and "West" in actions:
            actions.remove("West")

        start = time.time()

        path = self.getActionPlans(gameState)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()

        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        if len(path) > 0:
            action = path[0]
            path.remove(action)
        else:
            action = random.choice(actions)

        self.lastState = gameState
        
        return action

    def getActionPlans(self, gameState):
        self.path = self.aStarSearch(gameState)

        return self.path

    def getSuccessors(self, state, actionList, gameState):
        successors = []
        position = state

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitWall = self.walls[nextx][nexty]

            if not hitWall:
                nextState = (nextx, nexty)
                cost = self.getCostFunction(nextState, gameState)
                successors.append(((nextState, actionList), action, cost))
        return successors

    def getCostFunction(self, nextState, gameState):
        agentState = gameState.data.agentStates[self.index]
        isPacman = agentState.isPacman
        cost = 0
        invaders = self.getInvaders(gameState)
        ghosts = self.getGhosts(gameState)
        invaderPos = [a.getPosition() for a in invaders]
        ghostPos = [a.getPosition() for a in ghosts]
        scaredGhostPos = []

        for index in self.getOpponents(gameState):
            if gameState.getAgentState(index).scaredTimer != 0:
                scaredGhostPos.append(gameState.getAgentState(index).getPosition())

        if isPacman:
            if nextState in ghostPos:
                if nextState not in scaredGhostPos:
                    cost = 9999
                else:
                    cost = 1
            else:
                cost = 1
        else:
            if gameState.getAgentState(self.index).scaredTimer == 0:
                if nextState in invaderPos:
                    cost = 0
                else:
                    cost = 1
            else:
                if nextState in invaderPos:
                    cost = 9999
                else:
                    cost = 1

            nextState_x, nextState_y = nextState
            if gameState.isOnRedTeam(self.index):
                midWidth = int((self.width)/2)
                if nextState_x >= midWidth:
                    cost += 9999
            else:
                midWidth = int((self.width)/2)+1
                if nextState_x <= midWidth:
                    cost += 9999


        return cost

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos2 = successor.getAgentState(self.index).getPosition()
        x, y = pos2
        pos2 = (int(x), int(y))

        if pos2 != util.nearestPoint(pos2):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # --------------------------Goal Setting------------------------#

    def isGoalState(self, gameState, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        currentPos = state
        goalPosition = self.getGoalPosition(gameState)

        if currentPos == goalPosition:
            isGoal = True
        else:
            isGoal = False

        return isGoal

    def getCurrentPos(self, gameState):
        currentPos = gameState.getAgentPosition(self.index)
        return currentPos

    def getGoalPosition(self, gameState):
        goalPosition = None
        return goalPosition

    # -----------------Getting Information From layout-------------------#

    def getInvaders(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        return invaders

    def getGhosts(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        return ghosts

    def getDefendingTarget(self, gameState):
        return self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)

    def defendingTargetChanges(self, gameState):
        if self.lastState == None:
            return False
        else:
            return len(self.getDefendingTarget(self.lastState)) == len(self.getDefendingTarget(gameState))

    def nearestGhostInfo(self, gameState):
        currentPos = self.getCurrentPos(gameState)
        invaders = self.getInvaders(gameState)
        minDistance = 9999
        if len(invaders) > 0:
            for a in invaders:
                distance = self.manhattonDistance(currentPos, a.getPosition())
                if distance <= minDistance:
                    minDistance = distance
                    pos = a.getPosition()
                    return pos, minDistance
        return None, -9999

    def getNearestHome(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        nearestHomeDist, nearestHome = min([(self.getMazeDistance(returnPos, myPos), returnPos)
                                            for returnPos in self.patrolPositions])
        return nearestHome

    def getPatrolPosition(self, gameState):
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        patrolPositions = []
        if self.red:
            centralX = int((width - 2) / 2)
        else:
            centralX = int(((width - 2) / 2) + 1)
        for i in range(1, height - 1):
          
          if not gameState.hasWall(centralX, i):

            patrolPositions.append((centralX, i))
        return patrolPositions


    def heuristic(self, state, gameState):

        h = {}
        hValue = 0

        goalPosition = self.getGoalPosition(gameState)

        if self.manhattonDistance(state, goalPosition) <= 6:
            hValue = self.getMazeDistance(state, goalPosition)
        else:
            hValue = self.manhattonDistance(state, goalPosition)

        return hValue

    def aStarSearch(self, gameState):
        """Search the node that has the lowest combined cost and heuristic first."""
        currentPosition = self.getCurrentPos(gameState)
        path = []

        currentPos = currentPosition
        priorityQueue = util.PriorityQueue()
        cost_so_far = {}
        priorityQueue.push((currentPos, []), 0)
        cost_so_far[currentPos] = 0

        while not priorityQueue.isEmpty():
            currentPos, actionList = priorityQueue.pop()

            if self.isGoalState(gameState, currentPos):
                path = actionList

            nextMoves = self.getSuccessors(currentPos, actionList, gameState)
            for nextNode in nextMoves:
                new_cost = cost_so_far[currentPos] + nextNode[2]
                if nextNode[0][0] not in cost_so_far:
                    cost_so_far[nextNode[0][0]] = new_cost
                    priorityQueue.push((nextNode[0][0], actionList + [nextNode[1]]),
                                       new_cost + self.heuristic(nextNode[0][0], gameState))
                elif new_cost < cost_so_far[nextNode[0][0]]:
                    priorityQueue.push((nextNode[0][0], actionList + [nextNode[1]]),
                                       new_cost + self.heuristic(nextNode[0][0], gameState))

        return path

    def manhattonDistance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        distance = abs(x2 - x1) + abs(y2 - y1)
        return distance

    def getNearestFood(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        foods = self.getFood(gameState).asList()
        oneStepFood = len([food for food in foods if self.getMazeDistance(myPos, food) == 1])
        twoStepFood = len([food for food in foods if self.getMazeDistance(myPos, food) == 2])
        passBy = (oneStepFood == 1 and twoStepFood == 0)
        if len(foods) > 0:
            return min([(self.getMazeDistance(myPos, food), food, passBy) for food in foods])
        return None, None, None

    def getNearestCapsule(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0:
            return min([(self.getMazeDistance(myPos, cap), cap) for cap in capsules])
        return -9999, None

    def updateDefendingPos(self, gameState):
        self.partrolPosition = self.getPatrolPosition(gameState)
        self.lenPP = len(self.patrolPositions)
        #print self.partrolPosition
        positionNumber = self.posNum % self.lenPP
        self.initialDefendingPos = self.getPatrolPosition(gameState)[positionNumber]
        #print self.initialDefendingPos
        self.posNum = self.posNum + 1




class OffensiveAgent(RandomNumberTeamAgent):

    def getGoalPosition(self, gameState):
        invaders = self.getInvaders(gameState)
        ghosts = self.getGhosts(gameState)
        ghostPos, distanceToGhost = self.nearestGhostInfo(gameState)
        foods = self.getFood(gameState).asList()
        distanceTofood, foodPos, passBy = self.getNearestFood(gameState)
        Agent = gameState.data.agentStates[self.index]
        dToCapsule, capsulePos = self.getNearestCapsule(gameState)

        if len(ghosts) == 0:
            if len(foods) > 0 and Agent.numCarrying <= (len(foods) / 9.0):
                goalPosition = foodPos
            else:
                goalPosition = self.getNearestHome(gameState)
        else:
            if self.ghostComing(gameState):
                if dToCapsule < (distanceToGhost / 2.0) and dToCapsule != -9999 and distanceToGhost != 9999:
                    goalPosition = capsulePos
                elif distanceToGhost >= 4:
                    goalPosition = foodPos
                else:
                    goalPosition = self.getNearestHome(gameState)
            else:
                if len(foods) > 0 and Agent.numCarrying <= (len(foods) / 9.0):
                    goalPosition = foodPos
                else:
                    goalPosition = self.getNearestHome(gameState)

        return goalPosition

    def ghostComing(self, gameState):
        for index in self.getOpponents(gameState):
            if gameState.getAgentState(index).scaredTimer == 0:
                return True
        return False

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        if gameState.isOnRedTeam(self.index) and "East" in actions:
            actions.remove("East")
        elif not gameState.isOnRedTeam(self.index) and "West" in actions:
            actions.remove("West")

        path = self.getActionPlans(gameState)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()

        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        if len(path) > 0:
            action = path[0]
            path.remove(action)
        else:
            action = random.choice(actions)

        self.lastState = gameState
        return action
