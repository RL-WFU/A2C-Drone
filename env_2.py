import random
import sys
from ICRSsimulator import *
import numpy as np
import os


class Env:
    def __init__(self, args, image):
        # Create simulator object and load the image
        self.args = args
        self.sim = ICRSsimulator(image)
        if self.sim.loadImage() == False:
            print("Error: could not load image")
            sys.exit(0)

        # Simulate classification of mining areas
        lower = np.array([50, 80, 70])
        upper = np.array([100, 115, 110])
        interestValue = 1  # Mark these areas as being of highest interest
        self.sim.classify('Mining', lower, upper, interestValue)

        # Simulate classification of forest areas
        lower = np.array([0, 49, 0])
        upper = np.array([90, 157, 138])
        interestValue = 0  # Mark these areas as being of no interest
        self.sim.classify('Forest', lower, upper, interestValue)

        # Simulate classification of water
        lower = np.array([40, 70, 47])
        upper = np.array([70, 100, 80])
        interestValue = 0  # Mark these areas as being of no interest
        self.sim.classify('Water', lower, upper, interestValue)

        # Number of rows and colums of the map at the finest scale of classification
        # Each (i,j) position in the map is a 1-D array of classification likelihoods
        # of length given by the number of classes
        self.totalRows = args.env_dims
        self.totalCols = args.env_dims
        self.sim.setMapSize(self.totalRows, self.totalCols)
        self.sim.createMap()

        self.sight_dims = args.sight_dim
        self.image_size = (self.sight_dims * 2 + 1) * (self.sight_dims * 2 + 1)


        self.num_actions = 4

        self.battery = 100
        self.visited = np.ones([self.totalRows, self.totalCols])
        self.cached_points = []

        self.start_row = self.sight_dims
        self.start_col = self.sight_dims

        self.row_position = self.start_row
        self.col_position = self.start_col

        # for clarity maybe set constants as reward weights up here?
        # if keeping track of exact rotation, a sharp turn could also have a penalty
        #DEFAULT: 100, 0, 30
        self.MINING_REWARD = 100
        self.FOREST_REWARD = 0
        self.WATER_REWARD = 30
        self.RETURN_REWARD = 50
        self.TIMESTEP_PENALTY = -1
        self.BATTERY_PENALTY = -1000
        self.HOVER_PENALTY = -1
        self.COVERED_REWARD = 1000

        self.sim.setDroneImgSize(2, 2)
        navMapSize = self.sim.setNavigationMap()

    def reset(self):
        # reset visited states
        self.visited = np.ones_like(self.visited)

        # reset battery to 100
        self.battery = 100

        # clear cached points however you would do that

        # randomize starting position
        self.start_row = random.randint(self.sight_dims,self.totalRows - self.sight_dims - 1)
        self.start_col = random.randint(self.sight_dims,self.totalCols - self.sight_dims - 1)

        self.row_position = self.start_row
        self.col_position = self.start_col

        return self.row_position, self.col_position


    def step(self, action, explore):
        self.done = False
        # need to add controls for reaching the edge of the region

        next_row = self.row_position
        next_col = self.col_position

        # These are overly simplified discrete actions, will want to make this continuous at some point
        if action == 0 and self.row_position < (self.totalRows - self.sight_dims - 1):  # Forward one grid #261 - imgsize + 1
            next_row = self.row_position + 1
            next_col = self.col_position
        elif action == 0:
            action=4

        elif action == 1 and self.col_position < (self.totalCols - self.sight_dims - 1):  # right one grid #243 - imgsize + 1
            next_row = self.row_position
            next_col = self.col_position + 1
        elif action == 1:
            action=4

        elif action == 2 and self.row_position > self.sight_dims + 1:  # back one grid
            next_row = self.row_position - 1
            next_col = self.col_position
        elif action == 2:
            action = 4

        elif action == 3 and self.col_position > self.sight_dims + 1:  # left one grid
            next_row = self.row_position
            next_col = self.col_position - 1
        elif action == 3:
            action = 4



        self.sim.setDroneImgSize(self.sight_dims, self.sight_dims)
        navMapSize = self.sim.setNavigationMap()

        classifiedImage = self.sim.getClassifiedDroneImageAt(next_row, next_col)

        self.row_position = next_row
        self.col_position = next_col

        #self.battery_loss(action, last_action)
        #print("Battery:", self.battery)

        pursuit_reward, explore_reward = self.get_reward(classifiedImage, action, explore)

        self.visited_position()

        if action == 4:
            hover = 1
        else:
            hover = 0

        # End if battery dies
        # To end if returns to start position after so many steps add:
        # or (self.row_position == self.start_row and self.col_position == self.start_col and self.battery < 80)
        # (right now I am just using battery level since it decreases per time step but we can make this more sophisticated)
        if self.battery <= 0 or (self.row_position == self.start_row and self.col_position == self.start_col and self.battery < 80):
            self.done = True

        succ_visits = self.getVisitedBinaries()
        return classifiedImage, pursuit_reward, explore_reward, self.done, succ_visits, hover

    def get_reward(self, classifiedImage, action, explore):
        # decompose state for probabilities, then multiply for given rewards
        # are the rewards only for the exact box we're in or added for the entire range?
        # I kind of think it makes the most sense to reward only for the current spot
        # but it can see the other spots and know to go in that direction? I'm not sure exactly how that would work though

        #NEED TO DECIDE WHAT TO MAKE EXPLORE REWARD
        if self.done:
            covered = self.calculate_covered()
        else:
            covered = 0



        # If we don't want just the one position, then we can iterate over classifiedImage, adding all of each class
        mining_prob = classifiedImage[self.sight_dims, self.sight_dims,0]
        forest_prob = classifiedImage[self.sight_dims, self.sight_dims,1]
        water_prob = classifiedImage[self.sight_dims, self.sight_dims,2]

        #print(mining_prob, forest_prob, water_prob)
        hover = False
        if action == 4:
            hover = True

        pursuit_reward = mining_prob*self.MINING_REWARD + forest_prob*self.FOREST_REWARD + water_prob*self.WATER_REWARD
        pursuit_reward = pursuit_reward*self.visited[self.row_position, self.col_position]
        pursuit_reward += self.TIMESTEP_PENALTY + self.HOVER_PENALTY*hover




        explore_reward = self.visited[self.row_position, self.col_position]
        explore_reward += self.HOVER_PENALTY*hover


        return pursuit_reward, explore_reward

    # Maybe make an Agent class for battery, visited points, cached points, ect

    def calculate_covered(self):
        covered = 0
        for i in range(self.totalRows):
            for j in range(self.totalCols):
                if self.visited[i][j] < 1:
                    covered += 1

        percent_covered = covered / (self.totalCols * self.totalRows)
        return percent_covered

    def visited_position(self):
        # Two options: either count just the current, or count everything in it's field of vision
        self.visited[self.row_position, self.col_position] = 0

        for i in range(4):
            for j in range(4):
                self.visited[self.row_position + i - self.sight_dims, self.col_position + j - self.sight_dims] *= .7

    def plot_visited(self, episode):
        #plt.imshow(self.visited[:, :], cmap='gray', interpolation='none')
        #plt.title("Drone Path")
        #plt.show()
        plt.title("Drone Path Ep {}".format(episode))
        plt.imsave(fname=os.path.join(self.args.save_dir, 'Drone Path Ep {}.png'.format(episode)), arr=self.visited[:, :], cmap='gray')
        plt.clf()

    def getClassifiedDroneImage(self):
        self.sim.setDroneImgSize(self.sight_dims, self.sight_dims)
        navMapSize = self.sim.setNavigationMap()
        image = self.sim.getClassifiedDroneImageAt(self.row_position, self.col_position)
        return image

    def total_mining(self):
        mining = self.sim.ICRSmap[:, :, 0]

        mining_positions = []
        for i in range (self.totalRows - self.sight_dims):
            for j in range(self.totalCols - self.sight_dims):
                if mining[i][j] > 0 and i >= self.sight_dims and j >= self.sight_dims:
                    mining_positions.append([i, j])

        return mining_positions

    def getVisitedBinaries(self):
        successor_visits = np.zeros(4)
        down = [self.row_position+1 if self.row_position < self.args.env_dims-1 else self.row_position, self.col_position]
        right = [self.row_position, self.col_position+1 if self.col_position < self.args.env_dims-1 else self.col_position]
        up = [self.row_position-1 if self.row_position > 0 else self.row_position, self.col_position]
        left = [self.row_position, self.col_position-1 if self.col_position > 0 else self.col_position]
        #successor_visits[0] = 1 if self.visited[down[0], down[1]] == 0 else 0
        #successor_visits[1] = 1 if self.visited[right[0], right[1]] == 0 else 0
        #successor_visits[2] = 1 if self.visited[up[0], up[1]] == 0 else 0
        #successor_visits[3] = 1 if self.visited[left[0], left[1]] == 0 else 0

        successor_visits[0] = self.visited[down[0], down[1]]
        successor_visits[1] = self.visited[right[0], right[1]]
        successor_visits[2] = self.visited[up[0], up[1]]
        successor_visits[3] = self.visited[left[0], left[1]]

        return successor_visits

    def miningCovered(self):
        mining = self.sim.ICRSmap[:, :, 0]
        mining_positions = []
        mining_covered = []
        for i in range(self.totalRows):
            for j in range(self.totalCols):
                if mining[i, j] > 0.3:
                    mining_positions.append([i, j])
                    if self.visited[i, j] < 1:
                        mining_covered.append([i, j])

        total_mining = len(mining_positions)
        total_mining_covered = len(mining_covered)

        if total_mining == 0:
            percent_covered = .3
        else:
            percent_covered = total_mining_covered/total_mining

        return percent_covered, total_mining, total_mining_covered

    def waterCovered(self):
        water = self.sim.ICRSmap[:, :, 2]
        water_positions = []
        water_covered = []
        for i in range(self.totalRows):
            for j in range(self.totalCols):
                if water[i, j] > 0.3:
                    water_positions.append([i, j])
                    if self.visited[i, j] < 1:
                        water_covered.append([i, j])

        total_water = len(water_positions)
        total_water_covered = len(water_covered)
        if total_water == 0:
            percent_covered = .3
        else:
            percent_covered = total_water_covered / total_water

        return percent_covered


    def forestCovered(self):
        forest = self.sim.ICRSmap[:, :, 1]
        forest_positions = []
        forest_covered = []
        for i in range(self.totalRows):
            for j in range(self.totalCols):
                if forest[i, j] > 0.3:
                    forest_positions.append([i, j])
                    if self.visited[i, j] < 1:
                        forest_covered.append([i, j])

        total_forest = len(forest_positions)
        total_forest_covered = len(forest_covered)

        if total_forest == 0:
            percent_covered = .3
        else:
            percent_covered = total_forest_covered / total_forest

        return percent_covered

    def totalCovered(self):
        covered = 0
        for i in range(self.totalRows):
            for j in range(self.totalCols):
                if self.visited[i, j] < 1:
                    covered += 1

        percent_covered = covered / (self.totalCols * self.totalRows)
        return percent_covered



    '''
    def save_cached_point(self, row, col):
        cached_point = (row, col)
        self.cached_points.append(cached_point)
    def get_cached_point(self):
        # calculate distance from current position to all current cached points (separate function?) and return the closest one
        # I think it will be better if it can learn when to use cached points rather than hard coding,
        # but I'm not exactly sure how to set that up at this point
        # Consider comparing to just using GRU
        cached_point = self.cached_points[0]
        return cached_point
    '''
