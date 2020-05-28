import unittest
from ICRSsimulator import *

class ICRSsimulatorTest(unittest.TestCase):
	def setUp(self):
		self.sim = ICRSsimulator('testImage.png')
		self.sim.loadImage()

	def test_loadImage(self):
		sim = ICRSsimulator('testImage.png')
		self.assertEqual(sim.loadImage(), True)

	def test_classify(self):
		lower = np.array([250, 250, 0])   # color scheme is BGR
		upper = np.array([255, 255, 255])
		interestValue = 0
		self.sim.classify('Water', lower, upper, interestValue)

		lower = np.array([0, 250, 0])
		upper = np.array([0, 255, 0])
		interestValue = 0
		self.sim.classify('Forest', lower, upper, interestValue)

		lower = np.array([0, 250, 250])
		upper = np.array([255, 255, 255])
		interestValue = 0
		self.sim.classify('Mining', lower, upper, interestValue)
		self.assertEqual(self.sim.numberOfClasses(), 3)

	def test_createMap(self):
		rows = 6
		cols = 8
		self.sim.setMapSize(rows, cols)
		self.sim.createMap()
		self.sim.showMap()


if __name__ == '__main__':
    unittest.main()