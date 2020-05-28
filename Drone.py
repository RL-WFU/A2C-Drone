import threading
import multiprocessing
import numpy as np
from queue import Queue
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import sys

from ICRSsimulator import *


#THIS IS JUST A MODEL USED FOR FINDING DIMENSIONS
test_model = keras.Sequential()
test_model.add(layers.Dense(256, activation='relu'))
test_model.add(layers.Dense(64, activation='relu'))
test_model.add(layers.Dense(32, activation='relu'))
test_model.add(layers.Dense(4))

#Try feeding action into GRU to output (4,1)


class build_model(tf.keras.Model):
    def __init__(self, act_dimensions):
        super(build_model, self).__init__()
        self.act_dims = act_dimensions

        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')

        #How do I feed in the previous timestep from GRU and the previous action?
        self.GRU = layers.GRU(32, activation='relu')

        #self.mean = layers.Dense(self.act_dims, activation='tanh') FOR CONTINUOUS ACTION SPACE
        #self.variance = layers.Dense(self.act_dims, activation='softplus')

        self.logits = layers.Dense(4) #FOR DISCRETE ACTION SPACE
        self.value = layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        y = self.dense2(x)

        #Do I need to specify additional inputs for the GRU layer?
        gru = self.GRU(y)

        #means = self.mean(gru) #We don't use these yet because discrete actions in sim
        #variances = self.variance(gru)

        logits = self.logits(gru)

        value = self.value(gru)

        return logits, value


class Map:
    def __init__(self, numRows, numCols, img_name, sightDim):
        self.rows = numRows
        self.cols = numCols
        self.imgName = img_name
        self.navMap = ICRSsimulator(img_name)
        if self.navMap.loadImage() == False:
            print("Error: could not load image")
            sys.exit(0)
        self.navMap.setMapSize(numRows, numCols)

        self.sightDim = sightDim

        self.build_classification()

        self.navMap.createMap()

        self.init_drone()

    def build_classification(self):
        lower = np.array([50, 80, 70])
        upper = np.array([100, 115, 110])
        interestValue = 1
        self.navMap.classify('Mining', lower, upper, interestValue)

        lower = np.array([0, 49, 0])
        upper = np.array([90, 157, 138])
        interestValue = 0
        self.navMap.classify('Forest', lower, upper, interestValue)

        lower = np.array([40, 70, 47])
        upper = np.array([70, 100, 80])
        self.navMap.classify('Water', lower, upper, interestValue)

    def init_drone(self):
        self.navMap.setDroneImgSize(self.sightDim, self.sightDim)
        self.navMap.setNavigationMap()

    def get_classified_image_at_point(self, x, y):
        return self.navMap.getClassifiedDroneImageAt(x, y)


class MasterDrone:
    def __init__(self, sightDim, mapName, test):
        self.sightSize = sightDim

        self.map = Map(268, 250, mapName, self.sightSize)

        if test:
            self.global_model = test_model

        else:
            self.global_model = build_model(1)

        randomImage = self.map.get_classified_image_at_point(np.random.randint(0, 263), np.random.randint(0, 245))

        #CHANGES ARRAY FROM 6x6x3 to 36x3
        #(36, 3)
        #(1, (36, 3))
        randomImage = np.reshape(randomImage, (self.sightSize ** 2, 3))


        self.global_model(tf.convert_to_tensor(randomImage, dtype='float32'))

        self.global_model.summary()


agent = MasterDrone(6, 'sample12x12km2.jpg', True)


#Potentially look at other papers of UAV motion. Those have a fixed goal to reach, but that should
#only be affected by reward

