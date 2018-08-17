import numpy as np
import cv2
import pandas as pd
import os.path
import os
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import minimize
from .qContours import drawPoints
from skimage.segmentation import active_contour
import math
import sys
# Euclidean distance.
def euc_dist(pt1,pt2):
    return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))

def _c(ca,i,j,P,Q):
    sys.setrecursionlimit(2500)
    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        #import pdb; pdb.set_trace()
        ca[i,j] = euc_dist(P[0],Q[0])
    elif i > 0 and j == 0:
        ca[i,j] = max(_c(ca,i-1,0,P,Q),euc_dist(P[i],Q[0]))
    elif i == 0 and j > 0:
        ca[i,j] = max(_c(ca,0,j-1,P,Q),euc_dist(P[0],Q[j]))
    elif i > 0 and j > 0:
        ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),euc_dist(P[i],Q[j]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]

""" Computes the discrete frechet distance between two polygonal lines
Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
P and Q are arrays of 2-element arrays (points)
"""
def frechetDist(P,Q):
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    P = np.array(P)
    return _c(ca,len(P)-1,len(Q)-1,P,Q)


def hausdorffDistance(ptsA, ptsB):

    hdAB = directed_hausdorff(ptsA, ptsB)[0]
    hdBA = directed_hausdorff(ptsB, ptsA)[0]

    hd = np.max([hdAB, hdBA])

    return hd

def getError(UPts, UContour):

    StackedContours = np.squeeze(UContour)

    # blank_image = np.zeros((1200, 1600, 3), np.uint8)
    # blank_image = drawPoints(blank_image, np.squeeze(UContour),[255,255,255], 1)
    # cv2.imshow("Contours", blank_image)
    # cv2.waitKey(0)

    Err = 0
    Cpts = np.array([0, 0])
    Cidx = []
    for j in range(len(UPts)):
        DiffX = StackedContours[:,0] - UPts[j,0]
        DiffY = StackedContours[:,1] - UPts[j,1]

        Dist = np.square(DiffX) + np.square(DiffY)
        Err = Err + np.min(Dist)
        midx = np.argmin(Dist)

        Cpts = np.vstack(((Cpts), (StackedContours[midx])))
        Cidx.append(midx)


    Cpts = np.delete(Cpts, 0, 0)
    Err = (Err/(len(UPts)))

    #print(UPts)
    #print(Cpts)
    return Err,Cpts,Cidx

def findCentroid(Pts):

    l = Pts.shape[0]
    mX = np.sum(Pts[:,0])/l
    sY = np.sum(Pts[:,1])/l

    return mX, sY

def findContourCenter(Contour):


    M = cv2.moments(Contour)
    cX = np.int16(M["m10"] / M["m00"])
    cY = np.int16(M["m01"] / M["m00"])

    return cX, cY

def lineDist(x0, y0, x2, m, c):

    y2 = m*x2 + c

    dist = np.sqrt((x0 - x2)**2 + (y0 - y2)**2)

    return dist

def balloonFit(UPts_org, UCtr, UContour):       #Algorithm to push points away from centroid

    UContour = np.squeeze(UContour)

    # blank_image = np.zeros((1200, 1600, 3), np.uint8)
    # blank_image = drawPoints(blank_image, np.squeeze(UContour), [255, 255, 255], 1)
    #
    # blank_image = drawPoints(blank_image, UCtr, [255, 0, 0], 5)

    UPts = UPts_org.copy()

    for i in range(len(UPts)):

        X = tuple((UPts[i, 0], UPts[i, 1]))
        cval = cv2.pointPolygonTest(UContour, X, False)

        # if cval <= 0:  # If point lies along or outside a contour
        #     blank_image2 = drawPoints(blank_image.copy(), UPts[i], [0, 0, 255], 5)
        # else:
        #     blank_image2 = drawPoints(blank_image.copy(), UPts[i], [0, 255, 0], 5)
        x2 = []; y2 = []

        while cval > 0:

            l = 5
            x0 = UCtr[0,0]
            y0 = UCtr[0,1]
            x1 = UPts[i,0]
            y1 = UPts[i,1]

            if x0==x1:              #If points are vertical...
                x2 = x0
                if y1 > y0:
                    y2 = y1 + l
                elif y1 < y0:
                    y2 = y1 - l
            elif y0==y1:            #If points are horizontal...
                y2 = y0
                if x1 > x0:
                    x2 = x1 + l
                elif x1 < x0:
                    x2 = x1 - l

            else:
                m =  np.float(np.float((y1 - y0))/np.float((x1 - x0)))
                C = y0 - m*x0

                a = m**2 + 1
                b = 2*m*C - 2*m*y1 - 2*x1
                c = C**2 + y1**2 + x1**2 - 2*C*y1 - l**2

                d = (b ** 2) - (4 * a * c)

                sol1 = (-b - np.sqrt(d)) / (2 * a)
                sol2 = (-b + np.sqrt(d)) / (2 * a)

                d1 = lineDist(x0, y0, sol1, m, C)
                d2 = lineDist(x0, y0, sol2, m, C)

                if(d1 > d2):
                    x2 = sol1
                else:
                    x2 = sol2

                y2 = m*x2 + C

            UPts[i,0] = np.int16(np.round(x2))
            UPts[i,1] = np.int16(np.round(y2))

            X = tuple((UPts[i, 0], UPts[i, 1]))
            cval = cv2.pointPolygonTest(UContour, X, False)

            # blank_image2 = drawPoints(blank_image.copy(), UPts[i], [0, 255, 0], 5)

            # cv2.imshow("Balloon2", blank_image2)
            # cv2.waitKey(100)

    # blank_image = drawPoints(blank_image, UPts, [0, 255, 0], 5)
    # cv2.imshow("Balloon", blank_image)
    # cv2.waitKey(0)

    return UPts


def snakeFit(UPts, UContour):

    blank_image = np.zeros((1200, 1600, 3), np.uint8)

    #drawPoints(blank_image, np.squeeze(UContour), [255, 255, 255], 0)
    cv2.drawContours(blank_image, UContour, -1, [255, 255, 255], 5)
    #cv2.fillPoly(blank_image, pts=[np.squeeze(UContour)], color=(255, 255, 255))
    snake = active_contour(blank_image, np.array(UPts), bc='free', alpha=0.015, beta=10, gamma=0.001)
    drawPoints(blank_image, UPts, [0, 0, 255], 5)
    drawPoints(blank_image, np.int16(snake), [0, 255, 0], 5)

    #import pdb;
    #pdb.set_trace()

    drawPoints(blank_image, np.squeeze(UContour), [255, 255, 255], 0)
    cv2.imshow("Temp", blank_image)
    cv2.waitKey(0)


def computeAffineMatrix(params):

    sx = params[0]
    sy = params[1]
    shx = params[2]
    shy = params[3]
    theta = params[4]
    b0 = params[5]
    b1 = params[6]

    b = np.array((b0 , b1))

    TMatrix = np.matrix([[1, 0, b[0]], [0, 1, b[1]], [0, 0, 1]])
    RMatrix = np.matrix([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    ScMatrix = np.matrix([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    ShMatrix = np.matrix([[1, shx, 0], [shy, 1, 0], [0, 0, 1]])  # np.identity(3)

    M = RMatrix * ShMatrix * ScMatrix# * TMatrix

    return M

def getFit(x, vals):


    M = computeAffineMatrix(x)

    IPts0 = vals[0]

    IPts = np.append(IPts0[:,0] - vals[1][0], IPts0[:,1] - vals[1][1],axis=1)
    IPts2 = np.append(IPts, np.ones([IPts.shape[0], 1], dtype=np.int32), axis=1)

    UPts2 = np.transpose(M * np.transpose(IPts2))
    UPts = np.transpose((UPts2[:, 0] + vals[1][0], UPts2[:, 1] + vals[1][1]))
    UPts = np.matrix(np.int16(UPts))

    Err, Cpts, Cidx = getError(UPts, vals[2])

    return Err

def bestFit(UPts, UCentroid, UContour, thresh=3):

    BPts = balloonFit(UPts, UCentroid, UContour)
    _, BPts, _ = getError(BPts, UContour)

    _, CPts, _ = getError(UPts, UContour)

    Best = np.zeros(BPts.shape, np.int16)

    for i in range(len(BPts)):

        distB = np.sqrt(np.sum(np.square(BPts[i] - UPts[i])))
        distC = np.sqrt(np.sum(np.square(CPts[0] - UPts[0])))

        if(distB > thresh*distC):

            Best[i] = CPts[i]

        else:

            Best[i] = BPts[i]


    return Best


def getFitClosest(x, IPts0, Tx, Ty, UContour, UCentroid):

    M = computeAffineMatrix(x)
    #import pdb; pdb.set_trace()
    #IPts = np.append(IPts0[:,0] - vals[1][0], IPts0[:,1] - vals[1][1],axis=1)
    IPts = np.column_stack((IPts0[:,0] - Tx, IPts0[:,1] - Ty))
    IPts2 = np.append(IPts, np.ones([IPts.shape[0], 1], dtype=np.int32), axis=1)

    UPts2 = np.int16(1*np.transpose(M * np.transpose(IPts2)))
    UPts = np.transpose((UPts2[:, 0] + UCentroid[0, 0], UPts2[:, 1] + UCentroid[0, 1]))
    UPts = np.matrix(np.int16(UPts))

    UPts = bestFit(UPts, UCentroid, UContour)
    Err, Cpts, Cidx = getError(UPts, UContour)
    # import pdb;
    # pdb.set_trace()
    return Err, Cpts, UPts, Cidx

def plotFit(Iimg, Uimg, x, IPts0, Tx, Ty, extra=[], col = [0, 0, 255]):

    M = computeAffineMatrix(x)

    #IPts = np.append(IPts0[:,0] - vals[1][0], IPts0[:,1] - vals[1][1],axis=1)
    IPts = np.column_stack((IPts0[:, 0] - Tx, IPts0[:, 1] - Ty))
    IPts2 = np.append(IPts, np.ones([IPts.shape[0], 1], dtype=np.int32), axis=1)

    UPts2 = np.transpose(M * np.transpose(IPts2))
    UPts = np.transpose((UPts2[:, 0] + Tx, UPts2[:, 1] + Ty))
    UPts = np.matrix(np.int16(UPts))

    img1 = drawPoints(Iimg, IPts0, [255, 0, 0], 3)

    #img2 = drawPoints(Uimg, UPts, col, 3)
    if (len(extra) > 0):
        img2 = drawPoints(Uimg, extra, [0,255,0], 3)
    else:
        img2 = drawPoints(Uimg, UPts, col, 3)

    cv2.namedWindow("Ideal", cv2.WINDOW_NORMAL)
    cv2.namedWindow("User", cv2.WINDOW_NORMAL)
    cv2.imshow("Ideal", img1)
    cv2.imshow("User", img2)
    cv2.imwrite("UserOut.png", img2)
    cv2.waitKey(0)

    #print UPts

def guessAffineParameters(UserData, TempData):

    sx = UserData[1][0] / TempData[1][0]
    sy = UserData[1][1] / TempData[1][1]
    shx = 0
    shy = 0
    theta = (np.pi/180) * (UserData[2] - TempData[2])
    b = np.array((UserData[0][0] - TempData[0][0] , UserData[0][1] - TempData[0][1]))
    x0 = [sy, sx, shx, shy, theta, b[0], b[1]]

    return x0

