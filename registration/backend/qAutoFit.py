from .qContours import getUserContourFromImage, getTemplateContourFromImage, createMarkers, drawPoints
from .qContourTools import getError, getFitClosest, guessAffineParameters, computeAffineMatrix, hausdorffDistance, frechetDist, findContourCenter, balloonFit
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
