from qAutoFit import qAutoFit
import numpy as np
from qContours import drawPoints
import cv2
import time
import glob
import os

def getFileNameList(dirname, ext = '*.*'):

    fullname = dirname + os.sep + ext
    allfiles = glob.glob(fullname)
    return allfiles

def AutoRegister(dirname, outdir, NMarkers):

    infiles = getFileNameList(dirname, '*.png')
    TemplateFile = 'Ideal_SC_C7I.png'
    TemplateCSV = TemplateFile + '.csv'

    for i in range(len(infiles)):

        UserFile = infiles[i]
        UserCSV = UserFile + '.csv'

        start_time = time.time()
        inst = qAutoFit(UserFile, TemplateFile, NMarkers, SegLength=1000, SegStride=400)
        autopts = inst.fitAll()
        end_time = time.time()
        print('Time elapsed:', end_time - start_time)

        drawPoints(inst.UserImage, autopts[0], [0, 255, 0], sz=5)
        drawPoints(inst.UserImage,autopts[1],[0, 255, 0],sz=5)
        outFile = outdir + os.path.basename(UserFile)
        cv2.imwrite(outFile, inst.UserImage)
        np.savetxt(UserCSV, np.vstack(autopts), delimiter=',', fmt='%4d')

    np.savetxt(TemplateCSV, np.vstack(inst.Template_primary_markers), delimiter=',', fmt='%4d')

NMarkers = [10, 15]
DirName = './In/'
OutDir = './Out/'
AutoRegister(DirName, OutDir, NMarkers)



# Display Points (Optional)
# drawPoints(inst.TemplateImage,inst.Template_primary_markers[0],[255, 0, 0],sz=5)
# drawPoints(inst.TemplateImage,inst.Template_primary_markers[1],[255, 0, 0],sz=5)

# drawPoints(inst.UserImage,autopts[0],[0, 255, 0],sz=5)
# drawPoints(inst.UserImage,autopts[1],[0, 255, 0],sz=5)

# cv2.imshow("Template", inst.TemplateImage)
# cv2.imshow("User", inst.UserImage)
# cv2.waitKey(0)



from qContours import getUserContourFromImage, getTemplateContourFromImage, createMarkers, drawPoints
from qContourTools import getError, getFitClosest, guessAffineParameters, computeAffineMatrix, hausdorffDistance, frechetDist, findContourCenter, balloonFit
import numpy as np
import cv2
import time

def testContour(contour, pts=[]):

    blank_image = np.zeros((1200, 1600, 3), np.uint8)

    cv2.drawContours(blank_image,contour,-1,[255, 255, 255], -1)

    cv2.imshow("test", blank_image)
    cv2.waitKey(0)

class qAutoFit:

    UFileName = []
    TFileName = []
    Initial_Affine_Matrix = []
    Final_Affine_Matrix = []
    UserEllipse = []
    TemplateEllipse = []
    UserImage = []
    TemplateImage = []
    UserContours = []
    TemplateContours = []
    Template_primary_markers = []
    Template_primary_markers_loc = []
    User_initial_marker_guess = []
    User_initial_centroid = []
    User_initial_marker_loc = []
    Demo_img = []
    SegmentLength = []
    SegmentStride = []
    VideoOut = []
    TemplateContourCenter = []
    UserContourCenter = []

    def __init__(self, UserFile, TemplateFile, NumMarkers, SegLength=1000, SegStride=100, optMarkerFile=[]):

        self.UFileName = []
        self.TFileName = []
        self.Initial_Affine_Matrix = []
        self.Final_Affine_Matrix = []
        self.UserEllipse = []
        self.TemplateEllipse = []
        self.UserImage = []
        self.TemplateImage = []
        self.UserContours = []
        self.TemplateContours = []
        self.Template_primary_markers = []
        self.Template_primary_markers_loc = []
        self.User_initial_marker_guess = []
        self.User_initial_centroid = []
        self.User_initial_marker_loc = []
        self.Demo_img = []
        self.SegmentLength = []
        self.SegmentStride = []
        self.VideoOut = []
        self.TemplateContourCenter = []
        self.UserContourCenter = []

        self.UFileName = UserFile
        self.TFileName = TemplateFile

        self.UserEllipse, self.UserImage, self.UserContours = getUserContourFromImage(self.UFileName)
        self.TemplateEllipse, self.TemplateImage, self.TemplateContours = getTemplateContourFromImage(self.TFileName)

        x0 = guessAffineParameters(self.UserEllipse, self.TemplateEllipse)
        self.Initial_Affine_Matrix = computeAffineMatrix(x0)

        TemplateEllipseCenter = [self.TemplateEllipse[0][0], self.TemplateEllipse[0][1]]


        if (optMarkerFile != []):
            self.loadMarkersFromFile(optMarkerFile, NumMarkers)

        for cselect in range(len(self.TemplateContours)):

            if(optMarkerFile==[]):
                self.Template_primary_markers, self.Template_primary_markers_loc = createMarkers(self.TemplateContours, NumMarkers)
            TemplateCentroid = np.array(findContourCenter(self.TemplateContours[cselect]))
            UserCentroid = np.array(findContourCenter(self.UserContours[cselect]))
            self.TemplateContourCenter.append(TemplateCentroid)
            self.UserContourCenter.append(UserCentroid)

            UCentroid = np.reshape(UserCentroid, (1, 2))#self.applyAffineTransform(self.Initial_Affine_Matrix, TemplateCentroid)

            #testContour(self.UserContours[1])

            Err, Cpts, Upts, Cidx = getFitClosest(x0, self.Template_primary_markers[cselect], TemplateEllipseCenter[0], TemplateEllipseCenter[1], self.UserContours[cselect], UCentroid)

            # test = self.UserImage.copy()
            # drawPoints(test, Upts, [0, 255, 0], 5)
            # drawPoints(test, UCentroid, [0, 0, 255], 5)
            # cv2.imshow("Test", test)
            # cv2.waitKey(0)

            self.User_initial_marker_loc.append(Cidx)
            self.User_initial_marker_guess.append(Upts)

        self.SegmentLength = SegLength
        self.SegmentStride = SegStride


    def loadMarkersFromFile(self, fn, NumMarkers):

        mpoints = np.loadtxt(fn, delimiter=',')
        cols = mpoints.shape[1]
        if cols>2:  #If 3 columns, ignore first column
            mpoints = np.int16(mpoints[:, 1:])

        init = 0
        for i in range(len(NumMarkers)):
            X = mpoints[init:NumMarkers[i]+init]
            Err, X, cidx = getError(X, self.TemplateContours[i])
            init = init + NumMarkers[i]
            self.Template_primary_markers.append(X)
            self.Template_primary_markers_loc.append(cidx)

        #import pdb; pdb.set_trace()

        # im1 = drawPoints(self.TemplateImage.copy(), mpoints, [255, 0, 0], sz=5)
        # im2 = drawPoints(self.TemplateImage.copy(), self.Template_primary_markers[0], [255, 0, 0], sz=5)
        # im2 = drawPoints(im2, self.Template_primary_markers[1], [255, 0, 0], sz=5)
        # cv2.imshow("Org Template", im1)
        # cv2.imshow("New Template", im2)
        # cv2.waitKey(0)


    def applyAffineTransform(self, TMatrix, Points):

        if len(Points.shape) == 1:
            Points = np.reshape(Points, (1, 2))

        TemplateEllipseCenter = [self.TemplateEllipse[0][0], self.TemplateEllipse[0][1]]
        t_Points = np.column_stack((Points[:, 0] - TemplateEllipseCenter[0], Points[:, 1] - TemplateEllipseCenter[1]))
        t_Points = np.append(t_Points, np.ones([t_Points.shape[0], 1], dtype=np.int32), axis=1)

        UPts2 = np.transpose(TMatrix * np.transpose(t_Points))
        UPts = np.transpose((UPts2[:, 0] + TemplateEllipseCenter[0], UPts2[:, 1] + TemplateEllipseCenter[1]))
        UPts = np.matrix(np.int16(UPts))

        return UPts


    def getTemplateSegments(self, L=100, TargetContour = 0):

        Segments = []
        for i in range(len(self.Template_primary_markers_loc[TargetContour])):

            start_index = np.int32(self.Template_primary_markers_loc[TargetContour][i] - L/2)
            end_index = np.int32(self.Template_primary_markers_loc[TargetContour][i] + L/2)
            seg = []

            if(end_index > np.squeeze(self.TemplateContours[TargetContour]).shape[0] - 1):

                end_index = end_index - ((self.TemplateContours[TargetContour]).shape[0])

            if(start_index < 0):

                start_index = ((self.TemplateContours[TargetContour]).shape[0]) + start_index

            if start_index > end_index:

                seg1 = np.squeeze(self.TemplateContours[TargetContour][start_index:])
                seg2 = np.squeeze(self.TemplateContours[TargetContour][:end_index])

                if len(seg1.shape) == 1:
                    seg1 = np.reshape(seg1, (1, 2))

                if len(seg2.shape) == 1:
                    seg2 = np.reshape(seg2, (1, 2))

                seg = np.concatenate([seg1, seg2])
            else:
                seg = np.squeeze(self.TemplateContours[TargetContour][start_index:end_index])

            Segments.append(seg)

        return Segments

    def getUserSegment(self, seg_center_id, L=100, TargetContour = 0):

        start_index = np.int32(seg_center_id - L/2)
        end_index = np.int32(seg_center_id + L/2)

        seg = []

        if (end_index > np.squeeze(self.UserContours[TargetContour]).shape[0] - 1):
            end_index = end_index - ((self.UserContours[TargetContour]).shape[0])

        if (start_index > np.squeeze(self.UserContours[TargetContour]).shape[0] - 1):
            start_index = start_index - ((self.UserContours[TargetContour]).shape[0])

        if (start_index < 0):
            start_index = ((self.UserContours[TargetContour]).shape[0]) + start_index

        if (end_index < 0):
            end_index = ((self.UserContours[TargetContour]).shape[0]) + end_index

        if start_index > end_index:

            seg1 = np.squeeze(self.UserContours[TargetContour][start_index:])
            seg2 = np.squeeze(self.UserContours[TargetContour][:end_index])

            if len(seg1.shape)==1 :
                seg1 = np.reshape(seg1, (1,2))

            if len(seg2.shape)==1 :
                seg2 = np.reshape(seg2, (1,2))

            seg = np.concatenate([seg1, seg2])

        else:
            seg = np.squeeze(self.UserContours[TargetContour][start_index:end_index])

        return seg

    def fitSegment(self,  segment, temp_center, target_center_id, SegLength = 200, TargetContour=0):

        if target_center_id< 0:
            target_center_id = ((self.UserContours[TargetContour]).shape[0]) + target_center_id

        if target_center_id > np.squeeze(self.UserContours[TargetContour]).shape[0] - 1:
            target_center_id = target_center_id - ((self.UserContours[TargetContour]).shape[0])

        target_center = self.UserContours[TargetContour][target_center_id]
        aSeg = segment - (temp_center - target_center)

        uSeg = self.getUserSegment(target_center_id, SegLength, TargetContour)

        err = hausdorffDistance(aSeg, uSeg)
        #
        # temp_img = self.Demo_img.copy()
        # im2 = drawPoints(temp_img, aSeg, [0, 0, 255], 2)
        # im2 = drawPoints(im2, uSeg, [255, 0, 0], 2)
        #
        #
        # cv2.imshow("Temp", im2)
        # cv2.waitKey(1)

        return err


    def fitAll(self):

        OutPts = []
        self.Demo_img = self.UserImage.copy()

        for TargetContour in range(len(self.UserContours)):

            Segs = self.getTemplateSegments(self.SegmentLength, TargetContour)
            new_pt_locs = []
            for k in range(len(Segs)):
            #k=5
            #if (TargetContour==1) & (k==5):

                aSeg = self.applyAffineTransform(self.Initial_Affine_Matrix, Segs[k])
                aSeg_center = self.applyAffineTransform(self.Initial_Affine_Matrix,
                                                        self.Template_primary_markers[TargetContour][k:k + 1])
                center_id = self.User_initial_marker_loc[TargetContour][k]

                min_err = np.inf
                min_idx = np.inf

                for i in range(center_id - self.SegmentStride, center_id + self.SegmentStride):
                    serr = self.fitSegment(aSeg, aSeg_center, i, self.SegmentLength, TargetContour)

                    if (serr < min_err):
                        min_err = serr
                        min_idx = i

                if min_idx < 0:
                    min_idx = ((self.UserContours[TargetContour]).shape[0]) + min_idx

                if min_idx > np.squeeze(self.UserContours[TargetContour]).shape[0] - 1:
                    min_idx = min_idx - ((self.UserContours[TargetContour]).shape[0])

                new_pt_locs.append(min_idx)

                target_center = self.UserContours[TargetContour][min_idx]
                aSeg_temp = aSeg - (aSeg_center - target_center)

                #im2 = drawPoints(self.Demo_img, aSeg_temp, [0, 0, 255], 2)
                #im2 = drawPoints(self.Demo_img, np.squeeze(self.UserContours[TargetContour][min_idx]), [0, 255, 0], 5)

            final_user_pts = np.squeeze(self.UserContours[TargetContour])[new_pt_locs]
            OutPts.append(final_user_pts)

        return OutPts








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











import numpy as np
import cv2
import pandas as pd
import os.path
import os
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import minimize
from qContours import drawPoints
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




