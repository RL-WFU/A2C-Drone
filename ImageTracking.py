import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np

#this function does nothing, used as a placeholder for the createTrackbar function
def nothing(x):
    pass

im = np.zeros((300,512,3), np.uint8)
cv2.namedWindow("Trackbars")

cv2.createTrackbar("R", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("G", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("B", "Trackbars", 0, 255, nothing)

cv2.setTrackbarPos("G", "Trackbars", 10)
cv2.setTrackbarPos("B", "Trackbars", 80)
cv2.setTrackbarPos("R", "Trackbars", 150)

lowerBound=np.array([33, 80, 44])
upperBound=np.array([102, 255, 255])

kernelOpen = np.ones((5,5))
kernelClose = np.ones((20,20))

#initialize camera object
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30

#Using the PiRGBArray(). This is a 3D array that allows us to read frames from the camera.
#Takes two arguments: first is the camera object, second is the resolution.
rawCapture = PiRGBArray(camera, size=(640, 480))

#capture_continuous function starts reading continuous frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port="True"):
    image = frame.array
    #Setting up the Color Recognition
    #Using HSV (Hue Saturation Value) method of thresholding
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    B = cv2.getTrackbarPos("B", "Trackbars")
    G = cv2.getTrackbarPos("G", "Trackbars")
    R = cv2.getTrackbarPos("R", "Trackbars")
    
    im[:] = [B, G, R]
    
    #Find the lower and upper limit of the color in HSV
    green = np.uint8([[[B, G, R]]])
    hsvGreen = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
    lowerBound = np.uint8([hsvGreen[0][0][0]-10,100,100])
    upperBound = np.uint8([hsvGreen[0][0][0]+10,255,255])
    
    #adjust the threshold of the HSV image for a range of each selected color.
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    result = cv2.bitwise_and(image , image , mask=mask)
    
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
    
    maskFinal = maskClose
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    for i in range(len(conts)):
        x,y,w,h=cv2.boundingRect(conts[i])
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255), 2)
    
    cv2.imshow("frame", image)
    #cv2.imshow("hsv", hsv)
    cv2.imshow("mask", maskFinal)
    #cv2.imshow("result", result)
    
    count = cv2.countNonZero(maskFinal)
    
    key = cv2.waitKey(1)
    rawCapture.truncate(0)

    if key == 27:
            break

cv2.destroyAllWindows()