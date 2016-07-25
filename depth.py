import cv2
import numpy as np
import sys
import matplotlib as mp

I_big = cv2.imread('image2.jpg')
D_big = cv2.imread('image3.jpg')
M_big = cv2.imread('image1.jpg')
I = cv2.resize(I_big, (0,0), fx=0.25, fy=0.25)
D = cv2.resize(D_big, (0,0), fx=0.25, fy=0.25)
M = cv2.resize(M_big, (0,0), fx=0.25, fy=0.25)
row, col, lay = np.shape(M)
print(D)
R = np.array([[1], [2], [3]])
for x in range (0, row):
    for y in range (0, col):
            pixel1 = M[x, y, 0]
            if (pixel1 == 255):
                z = D[x, y, 0]
                R = np.concatenate((R, np.array([[x],[y], [z]])), axis = 1)
r, c = np.shape(R)
Z = R[:, 1:c]
X = Z[0:2,:]
#svd

svdZ = np.linalg.svd(Z, full_matrices=False)
svdX = np.linalg.svd(X, full_matrices=False)
print(svdZ[0])
print(np.dot(np.diag(svdZ[1]),svdZ[2]))
rowz, colz = np.shape(Z)
rowx, colx = np.shape(X)

#mp.subplot(nrows=rowz, ncols = colz)
