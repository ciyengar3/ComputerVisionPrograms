# import the necessary packages
import cv2
import numpy as np

factor = 1


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)
# construct the argument parser and parse the arguments

# load the image, clone it, and setup the mouse callback function
cap = cv2.VideoCapture(0)
#turns on videocamera
ret, image = cap.read()
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(10000)

mean_color, sd_color = cv2.meanStdDev(roi)
avg_blue = mean_color[0]
avg_green = mean_color[1]
avg_red = mean_color[2]
sd_blue = sd_color[0]
sd_green = sd_color[1]
sd_red = sd_color[2]
#add conditionals
lower_blue = avg_blue - factor*sd_blue
lower_blue = lower_blue[0]
if lower_blue < 0:
    lower_blue = 0
lower_green = avg_green - factor*sd_green
lower_green = lower_green[0]
if lower_green < 0:
    lower_green = 0
lower_red = avg_red - factor*sd_red
lower_red = lower_red[0]
if lower_red < 0:
    lower_red = 0
higher_blue = avg_blue + factor*sd_blue
higher_blue = higher_blue[0]
if higher_blue > 255:
    higher_blue = 255
higher_green = avg_green + factor*sd_green
higher_green = higher_green[0]
if higher_green > 255:
    higher_green = 255
higher_red = avg_red + factor*sd_red
higher_red = higher_red[0]
if higher_red > 255:
    higher_red = 255
lower_color = np.uint8([lower_blue,lower_green,lower_red])
print(lower_color)
higher_color = np.uint8([higher_blue,higher_green,higher_red])
print(higher_color)
new_cap = cv2.VideoCapture(0)
#creates a videocapture object
ret = True
while(ret):
    #this captures frame by frame
    ret2, frame2 = cap.read()
    hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    #converts from BGR to HSV
    #defining the limits of the array(green color)
    mask = cv2.inRange(frame2, lower_color, higher_color)
    res = cv2.bitwise_and(frame2,frame2,mask = mask)
    #isolates the pixels from the mask
    cv2.imshow('frame2',frame2)
    cv2.imshow('result', mask)
    cv2.imshow('res',res)
    l = cv2.waitKey(100/3)
    if (l == ord('q')):
        cv2.destroyAllWindows()
        break
    #close and exit

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    #converts to black and white



cap.release()
cv2.destroyAllWindows()


