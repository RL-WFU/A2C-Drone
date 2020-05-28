
import sys
from ICRSsimulator import *
import numpy as np

# Create simulator object and load the image
sim = ICRSsimulator('sample12x12km2.jpg')
if sim.loadImage() == False:
	print("Error: could not load image")
	sys.exit(0)

# Simulate classification of mining areas
lower = np.array([50, 80, 70])
upper = np.array([100, 115, 110])
interestValue = 1	# Mark these areas as being of highest interest
sim.classify('Mining', lower, upper, interestValue)

# Simulate classification of forest areas
lower = np.array([0, 49, 0])
upper = np.array([90, 157, 138])
interestValue = 0	# Mark these areas as being of no interest 
sim.classify('Forest', lower, upper, interestValue)

# Simulate classification of water
lower = np.array([40, 70, 47])
upper = np.array([70, 100, 80])
interestValue = 0	# Mark these areas as being of no interest
sim.classify('Water', lower, upper, interestValue)

# Number of rows and colums of the map at the finest scale of classification
# Each (i,j) position in the map is a 1-D array of classification likelihoods
# of length given by the number of classes
rows = 268
cols = 250
sim.setMapSize(rows, cols) #We will divide the pixels of the image by these values. So
#this represents the number of cells in our grid of the image

# The map will contain 3 values per map element, simulating the
# likelihood of finding Mining, Forest, and Water, in that order
sim.createMap()

# Show the map with the classification results in separate images
sim.showMap()

# Set the size of the drone image in terms of number of elements in
# the map.

#If this number is the same as the map size, the drone can see the whole map.
#Currently, the drone can see a 6x6 area around its position. Increase this number
#to go higher
sim.setDroneImgSize(6, 6)

# Create the navigation map for the drone
navMapSize = sim.setNavigationMap()
print(navMapSize)

# Get sample drone image classifications
classifiedImage = sim.getClassifiedDroneImageAt(0,0)
print(classifiedImage.shape)



fig, axs = plt.subplots(1, len(classifiedImage.shape))
for k in range(0, len(classifiedImage.shape)):
	axs[k].imshow(classifiedImage[:,:,k], cmap = 'gray', interpolation = 'none')
plt.show()

classifiedImage = sim.getClassifiedDroneImageAt(navMapSize[0] - 1, navMapSize[1] - 1)
print(classifiedImage.shape)
fig, axs = plt.subplots(1, len(classifiedImage.shape))
for k in range(0, len(classifiedImage.shape)):
	axs[k].imshow(classifiedImage[:,:,k], cmap = 'gray', interpolation = 'none')
plt.show()

