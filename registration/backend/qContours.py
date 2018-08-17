import numpy as np
import cv2
import pandas as pd
import os.path
import os
from scipy.interpolate import splprep, splev

def sortContours(contours, hierarchy, athresh = 5, pthresh = 5, NContourMax= 3):

    outContours = []

    ParentList = hierarchy[0,:,3]

    outContours.append(np.asarray(contours[0], dtype=np.int32))  # First contour will always be outermost contour
    OArea = cv2.contourArea(contours[0])
    OPeri = cv2.arcLength(contours[0], True)
    for i in range(len(contours) - 1):

        pidx = i + 1;   #Set current parent index

        pmatches = np.array(np.where(ParentList == pidx)) # Find all child contours that have parent == pidx
        matches = np.array(pmatches[0].shape)[0]

        SumArea = 0;
        PArea = cv2.contourArea(contours[pidx])
        PPeri = cv2.arcLength(contours[pidx], True)

        # test = np.zeros((2000, 2000, 3), np.uint8)
        # cv2.drawContours(test, contours, pidx, [255, 255, 255], -1)
        # cv2.imshow("contours", test)
        # cv2.waitKey(0)

        #import pdb; pdb.set_trace()
        if PPeri >= ((100 - pthresh)/100.0)*OPeri:  #If sub-contour perimeter is comparable to outer contour then its likely a dud
            continue
        # else:
        #     test = np.zeros((2000, 2000, 3), np.uint8)
        #     cv2.drawContours(test, contours, pidx, [255, 255, 255], -1)
        #     cv2.imshow("contours", test)
        #     cv2.waitKey(0)


        for j in range(matches):
            SumArea = SumArea + cv2.contourArea(contours[pmatches[0][j]])

        if(SumArea == 0) & (PArea > (athresh/100.0)*OArea):                                             # If there are no children, add parent contour to list
            outContours.append(np.asarray(contours[pidx], dtype=np.int32))

        elif (SumArea > 0) & (SumArea < ((100 - athresh)/100.0)* PArea):  # If sum of children's area < % of parents, add parent contour to list
            outContours.append(np.asarray(contours[pidx], dtype=np.int32))
        #
        # if (pidx == 2):
        #     import pdb;
        #     pdb.set_trace()

    #import pdb; pdb.set_trace()
    final_contours = []
    #final_contours.append(outContours[0])
    outimg = np.zeros((2000, 2000, 1), np.uint8)
    outimg0 = np.zeros((2000, 2000, 1), np.uint8)
    kernel = np.ones((20, 20), np.uint8)
    im0 = cv2.drawContours(outimg0, outContours, 0, [255, 255, 255], -1)   #Outermost contour
    closing = cv2.morphologyEx(im0, cv2.MORPH_CLOSE, kernel)
    _, contours0, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    final_contours.append(contours0[0])

    # test = np.zeros((2000, 2000, 3), np.uint8)
    # drawPoints(test, np.squeeze(contours0[0]), [255, 255, 255],1)
    # cv2.imshow("Test", test)
    # cv2.waitKey(0)

    for i in range (NContourMax-1):

        im1 = cv2.drawContours(outimg, outContours, i+1, [255, 255, 255], -1)

    closing = cv2.morphologyEx(im1, cv2.MORPH_CLOSE, kernel)
    _, contours2, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_contours.append(contours2[0])

    return final_contours

def getUserContourFromImage(UserFile):

    img = cv2.imread(UserFile,0)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    _, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
    contours = sortContours(contours, hierarchy)
    #import pdb; pdb.set_trace()

    outimg = np.zeros((img.shape[0],img.shape[1], 3), np.uint8)

    im2 = cv2.drawContours(outimg,contours, 0, [255, 255, 255], -1)

    ellipse = cv2.fitEllipse(contours[0])
    #print ellipse
    im2 = cv2.ellipse(im2,ellipse,(0,255,0),2)

    # cv2.imshow("Temp", im2)
    # cv2.waitKey(0)

    img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours = SmoothenContours(contours)
    return ellipse, img2, contours


def getMask(img):

    lower_blue = np.array([100, 0, 0])
    #upper_blue = np.array([255, 150, 200])
    upper_blue = np.array([255, 200, 0])
    mask = cv2.inRange(img, lower_blue, upper_blue)
    return mask

def getTemplateContourFromImage(TemplateFile):

    img = cv2.imread(TemplateFile, cv2.IMREAD_COLOR)
    mask = getMask(img)
    img2 = cv2.bitwise_not(img)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)

    _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours = sortContours(contours, hierarchy)

    outimg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    outimg2 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    im2 = cv2.drawContours(outimg, contours, 3, [255, 255, 255], 1)

    ellipse = cv2.fitEllipse(contours[0])
    #im2 = cv2.ellipse(im2, ellipse, (0, 255, 0), 2)

    _, mcontours, hierarchy2 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #mcontours = sortContours(mcontours, hierarchy2)
    #import pdb; pdb.set_trace()
    #im3 = cv2.drawContours(outimg2, mcontours, 0, [255, 255, 255], 1)

    outContours = []

    outContours.append(np.asarray(contours[0], dtype=np.int32))
    outContours.append(np.asarray(mcontours[0], dtype=np.int32))

    outContours = SmoothenContours(outContours)
    #cv2.imshow("Temp", im3)
    #cv2.waitKey(0)


    return ellipse, img, outContours

def SmoothenContours(contours, minRes=1):

    # Credit : http://agniva.me/scipy/2016/10/25/contour-smoothing.html

    smoothened = []
    for contour in contours:
        x, y = contour.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x, y], u=None, s=100.0, per=1)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), 5000)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        smoothened.append(np.asarray(res_array, dtype=np.int32))

    # Overlay the smoothed contours on the original image
    #cv2.drawContours(original_img, smoothened, -1, (255, 255, 255), 2)
    return smoothened

def drawPoints(img, points, col=[0, 0, 255], sz=1):

    #import pdb; pdb.set_trace()
    if(sz==1):
        if(len(points.shape)>1):
            img[points[:, 1], points[:, 0]] = col
        else:
            img[points[1], points[0]] = col
        return img
    else:
        if (len(points.shape) > 1):
            for i in range(len(points)):
                cv2.circle(img, tuple((points[i, 0], points[i, 1])), sz, col, -1)
        else:
            cv2.circle(img, tuple((points[0], points[1])), sz, col, -1)
        return img


def getEquidistantPoints(contour, N):

    perimeter_of_contour = int(cv2.arcLength(contour, True))
    ctr_points = np.squeeze(contour)

    equidistant_length = int(perimeter_of_contour / N)

    X = []
    segment = 0
    diff = np.diff(ctr_points, axis=0)
    dist = np.sqrt((diff ** 2).sum(axis=1))

    for k in range(len(contour) - 1):

        segment = segment + dist[k]
        # import pdb; pdb.set_trace()
        if np.int16(segment) >= equidistant_length:
            X.append(k)
            segment = 0

    if(len(X) < N):
        X.append(len(contour)-1)

    return X

def createMarkers(contours,len_array):

    MarkerList = []
    MarkerPos = []
    #import pdb; pdb.set_trace()
    for i in range(len(contours)):
        X = getEquidistantPoints(contours[i], len_array[i])
        MarkerList.append(np.squeeze(contours[i])[X])
        MarkerPos.append(X)

    return MarkerList, MarkerPos