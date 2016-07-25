import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
higherbound = 5
stepsize = 10

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
# load the image, clone it, and setup the mouse callback function
image = cv2.imread("fig.png", 0)
cv2.imshow("fig", image)
clone = image.copy()
r, c = np.shape(image)
#print(r, c)
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
	cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()
#I = cv2.resize(roi, (0,0), fx=0.25, fy=0.25)

x1 = refPt[0][0]
x2 = refPt[1][0]
y1 = refPt[0][1]
y2 = refPt[1][1]
#print(refPt )
if (x2 > x1):
    higherx = x2
    lowerx = x1
else:
    lowerx = x2
    higherx = x1

if (y2 > y1):
    highery = y2
    lowery = y1
else:
    highery = y1
    lowery = y2
#print(lowerx, higherx, lowery, highery)
R = np.array([[1], [2], [3]])
counter = 0
total = 0
for x in range (lowerx, higherx + 1):
    for y in range (lowery, highery + 1):
        z = image[y, x]
        total += z
        counter += 1
        R = np.concatenate((R, np.array([[y],[x], [z]])), axis = 1)

avgValue = total / counter
print(avgValue)
r, c = np.shape(R)
#print(r, c)
X = R[:, 1:c]
print(X)
mean_value = np.mean(X, axis = 1)
covariance = np.cov(X)
eig_value, eig_vect = np.linalg.eigh(covariance)
eig_vect = np.fliplr(eig_vect)
det = np.linalg.det(eig_vect)
if (det < 0):
	eig_vect[:,2] = -eig_vect[:, 2]
#eig_value = np.fliplr(eig_value)

#print(np.shape(X))
#print(mean_value)
#print(covariance)
#print(eig_vect)
#print(np.linalg.det(eig_vect))
#print(eig_value)
#print(np.transpose(eig_vect))
#print(np.transpose(eig_vect).dot(-mean_value))

Xcam = np.transpose(eig_vect).dot(X)
T = np.matrix(np.transpose(eig_vect).dot(-mean_value))
print(np.transpose(T))
Xcam = Xcam + np.transpose(T)
#print(Xcam)
r, c = np.shape(Xcam)
#print(r, c)
F = np.array([[1], [2], [3]])
for i in range(0, c):
	if (Xcam[2, i] >= higherbound):
		F = np.concatenate((F, np.array([[Xcam[0, i]],[Xcam[1,i]], [Xcam[2,i]]])), axis = 1)
F = F[:, 1:c]


fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
#for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    #xs = randrange(n, 23, 32)
    #ys = randrange(n, 0, 100)
    #zs = randrange(n, zl, zh)
r1, c1 = np.shape(X)
r, c = np.shape(F)
#for x in range (0, c1):
	#ax1.scatter(X[0, x], X[1, x], X[2, x])

ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')

#plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    #xs = randrange(n, 23, 32)
    #ys = randrange(n, 0, 100)
    #zs = randrange(n, zl, zh)
r, c = np.shape(F)
for x in range (0, c1):
	ax.scatter(Xcam[0, x], Xcam[1, x], Xcam[2, x])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

