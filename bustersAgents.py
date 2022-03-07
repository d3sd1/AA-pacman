from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters


def printLineDataGlobal(self, gameState):
    securityDistance = 5  # Hardcoded security distance
    posStr = str(gameState.getPacmanPosition()[0]) + "," + str(gameState.getPacmanPosition()[1])
    unsafeGhostsStr = ""
    unsafeGhostsNearby = False
    for idx, distance in enumerate(gameState.data.ghostDistances):
        if distance <= securityDistance and distance is not None:
            unsafeGhostsStr += str(idx) + ";"
            unsafeGhostsNearby = True
    if unsafeGhostsNearby: unsafeGhostsStr = unsafeGhostsStr[:-1]  # remove last ;
    nearestDotDistanceStr = str(gameState.getDistanceNearestFood())
    # positionX,positionY,[ghostsNearby],nearestDotDistance,pacmanDirection
    # [ghostsNearby] -> can be NULL or ; separated idx of the ghosts
    # TODO -> scoreSiguiente al final

    pacman_next_dir = 'STOP'
    next_score = '0'
    return posStr + "," + nearestDotDistanceStr + "," + str(gameState.data.layout.getNumGhosts()) + "," + str(
        gameState.getScore()) + "," + str(len(gameState.data.layout.capsules)) + "," + gameState.data.agentStates[
               0].getDirection() + "," + pacman_next_dir + "," + next_score


class NullGraphics(object):
    "Placeholder for graphics"

    def initialize(self, state, isBlue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def printLineData(self, gameState):
        return printLineDataGlobal(self, gameState)

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def printLineData(self, gameState):
        return printLineDataGlobal(self, gameState)

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True,
                 elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        # for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        # self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def printLineData(self, gameState):
        return printLineDataGlobal(self, gameState)

    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)


from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''


class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def printLineData(self, gameState):
        return printLineDataGlobal(self, gameState)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:  move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal: move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:   move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def printLineData(self, gameState):
        return printLineDataGlobal(self, gameState)

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i + 1]]
        return Directions.EAST


class BasicAgentAA(BustersAgent):

    def printLineData(self, gameState):
        return printLineDataGlobal(self, gameState)

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ",
              [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print(gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman
        pacmanPos = gameState.getPacmanPosition()

        closestGhostDistance = None
        closestGhostPosition = None

        for idx, alive in enumerate(gameState.getLivingGhosts()):
            distance = None
            if alive:
                distance = gameState.data.ghostDistances[idx - 1]
            if distance is not None and (closestGhostDistance is None or distance < closestGhostDistance):
                closestGhostDistance = distance
                closestGhostPosition = gameState.getGhostPositions()[idx - 1]

        # Check what do we have to prioritize: in case we moved X axis, try to move Y before now, and viceversa

        pacmanDirection = gameState.data.agentStates[0].getDirection()
        print(pacmanDirection)
        # Last move was Y axis, priorize axis X (default)
        if (pacmanDirection is None or pacmanDirection is Directions.STOP) or (
                pacmanDirection is Directions.SOUTH or pacmanDirection is Directions.NORTH):
            if closestGhostPosition[0] < pacmanPos[0]:
                move = Directions.WEST
            elif closestGhostPosition[0] > pacmanPos[0]:
                move = Directions.EAST
            elif closestGhostPosition[1] > pacmanPos[1]:
                move = Directions.NORTH
            elif closestGhostPosition[1] < pacmanPos[1]:
                move = Directions.SOUTH

        if pacmanDirection is Directions.EAST or pacmanDirection is Directions.WEST:
            if closestGhostPosition[1] > pacmanPos[1]:
                move = Directions.NORTH
            elif closestGhostPosition[1] < pacmanPos[1]:
                move = Directions.SOUTH
            elif closestGhostPosition[0] < pacmanPos[0]:
                move = Directions.WEST
            elif closestGhostPosition[0] > pacmanPos[0]:
                move = Directions.EAST

        # Return move only if that's valid. If not, stop!
        if move in legal:
            return move

        # Prevent stuck stuff! Illegal movement detected. Alternate X / Y axis
        if closestGhostPosition[1] > pacmanPos[1]:
            move = Directions.NORTH
        elif closestGhostPosition[1] < pacmanPos[1]:
            move = Directions.SOUTH
        elif closestGhostPosition[0] < pacmanPos[0]:
            move = Directions.WEST
        elif closestGhostPosition[0] > pacmanPos[0]:
            move = Directions.EAST

        # Return move only if that's valid. If not, stop!
        if move in legal:
            return move

        # Stuck stuff -> try to move to another axis (note that it's not circular or biyective).
        # EAST -> NORTH; NORTH -> EAST; WEST -> SOUTH; SOUTH -> WEST
        if pacmanDirection is Directions.EAST:
            move = Directions.NORTH
        elif pacmanDirection is Directions.WEST:
            move = Directions.SOUTH
        elif pacmanDirection is Directions.NORTH:
            move = Directions.EAST
        elif pacmanDirection is Directions.SOUTH:
            move = Directions.WEST

        # Only legal moves ahead our stocastical movements.
        if move in legal:
            return move

        # In case we can't move left, move right. In case we can't move up, move down. If this does not work, map is wrong, so we just STOP. We may just wait.
        move = Directions.REVERSE[move]
        if move in legal:
            return move

        # As a desperation movement. Rand it! Break the walls!! This happens when we have tons of walls nearby at the beginning.

        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:  return Directions.WEST
        if (move_random == 1) and Directions.EAST in legal: return Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:   return Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal: return Directions.SOUTH

        # No way. Just stop. Wait for a beer
        return Directions.STOP
