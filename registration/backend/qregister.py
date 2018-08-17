# Warp.py
# Wali
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial import Delaunay
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from scipy import ndimage
import pandas as pd
import os.path
import os
from skimage import transform as tf


def LoadCSVFile(fname1):
    file_flag = os.path.isfile(fname1)
    if (file_flag == True):  # If file exists
        df = pd.read_csv(fname1, header=None)
        X = list(df[1])
        Y = list(df[2])
        Labels = list(df[0])
        Coords = list(zip(X, Y))
        return Labels, Coords
    else:
        return [], []


def getAffineMatrix(to_references, from_references, img=[], cols=1600, rows=1200):
    tform = tf.estimate_transform('affine', (from_references), (to_references))
    matrix_affine = np.array(tform.params.copy())[0:2, :]

    if (len(img) > 0):
        img_intermediate = cv2.warpAffine(img, matrix_affine, (cols, rows))
        return matrix_affine, np.array(img_intermediate)
    else:
        return matrix_affine, np.array([])


def applyAffineTransform(M, pts):
    vones = np.ones(len(pts))

    # import pdb; pdb.set_trace()
    if (len(pts.shape) > 1):
        pts_wones = np.concatenate((np.matrix(pts), np.transpose(np.matrix(vones))), axis=1)
    else:
        pts_wones = np.matrix(np.append(pts, 1))

    affine_pts = pts_wones * np.transpose(M)

    return affine_pts


def getRegisteredImage(ideal_references, image_references, img, alpha=1):
    if img.size == 0:
        print("Error: must supply image")
        return

    # height and width of the ideal image
    siz = img.shape
    height = siz[0]
    width = siz[1]

    xA = np.array(ideal_references)[:, 0]
    yA = np.array(ideal_references)[:, 1]
    xA = np.concatenate((xA, [1], [width], [width], [1]), axis=0)
    yA = np.concatenate((yA, [1], [1], [height], [height]), axis=0)

    # xB and yB are the lists of x and y coordinates of the control
    # points on the individual image
    xB = np.array(image_references)[:, 0]
    yB = np.array(image_references)[:, 1]
    xB = np.concatenate((xB, [1], [width], [width], [1]), axis=0)
    yB = np.concatenate((yB, [1], [1], [height], [height]), axis=0)

    xC = alpha * xA + (1 - alpha) * xB
    yC = alpha * yA + (1 - alpha) * yB

    # create an intermediate grid
    # warp the intermediate grid to source and target grid
    gx = np.linspace(1, width, width)
    gy = np.linspace(1, height, height)
    X, Y = np.meshgrid(gx, gy)

    points = np.array(np.transpose(np.matrix([xC, yC])))
    triC = Delaunay(points)

    xCB, yCB = warp(X, Y, xB, yB, xC, yC, triC)

    VCB = np.zeros(img.shape)
    dimg = np.double(img)

    for i in range(3):
        ff = ndimage.map_coordinates(dimg[:, :, i], (np.int32(yCB), np.int32(xCB)), output=None, order=3)
        VCB[:, :, i] = ff
    VCB = np.uint8(VCB)  # change VCB to a uint8 and save it to imgOut
    imgOut = VCB

    return imgOut


def warpCoords(query_pts, xA, yA, xB, yB, imsize):
    width = imsize[0]
    height = imsize[1]
    Bpts = np.transpose(np.array((xB, yB)))

    matrix_affine, tmp = getAffineMatrix(np.transpose([xB, yB]), np.transpose([xA, yA]), [], width, height)

    query_Affine = applyAffineTransform(matrix_affine, query_pts)
    B_Affine = applyAffineTransform(matrix_affine, Bpts)

    intermediate_xA = np.concatenate((np.transpose(xA), [1], [width], [width], [1]), axis=0)
    intermediate_yA = np.concatenate((np.transpose(yA), [1], [1], [height], [height]), axis=0)

    intermediate_xB = np.concatenate((B_Affine[:, 0].A1, [1], [width], [width], [1]), axis=0)
    intermediate_yB = np.concatenate((B_Affine[:, 1].A1, [1], [1], [height], [height]), axis=0)

    points = np.array(np.transpose(np.matrix([intermediate_xA, intermediate_yA])))
    triC = Delaunay(points)

    warped_X, warped_Y = warp(query_Affine[:, 0], query_Affine[:, 1], np.int32(intermediate_xA), np.int32(intermediate_yA), np.int32(intermediate_xB), np.int32(intermediate_yB), triC)

    return warped_X, warped_Y


def inTri(vx, vy, v0x, v0y, v1x, v1y, v2x, v2y):
    w1 = ((vx - v2x) * (v1y - v2y) - (vy - v2y) * (v1x - v2x)) / (
                (v0x - v2x) * (v1y - v2y) - (v0y - v2y) * (v1x - v2x) + np.spacing(1))
    w2 = ((vx - v2x) * (v0y - v2y) - (vy - v2y) * (v0x - v2x)) / (
                (v1x - v2x) * (v0y - v2y) - (v1y - v2y) * (v0x - v2x) + np.spacing(1))
    w3 = 1 - w1 - w2

    chk1 = (np.float32(w1 >= 0))
    chk2 = (np.float32(w1 <= 1))
    w1 = np.multiply(np.multiply(chk1, chk2), w1);
    w2 = np.multiply(np.multiply(chk1, chk2), w2);
    w3 = np.multiply(np.multiply(chk1, chk2), w3);

    chk1 = (np.float32(w2 >= 0))
    chk2 = (np.float32(w2 <= 1))
    w1 = np.multiply(np.multiply(chk1, chk2), w1);
    w2 = np.multiply(np.multiply(chk1, chk2), w2);
    w3 = np.multiply(np.multiply(chk1, chk2), w3);

    chk1 = (np.float32(w3 >= 0))
    chk2 = (np.float32(w3 <= 1))
    w1 = np.multiply(np.multiply(chk1, chk2), w1);
    w2 = np.multiply(np.multiply(chk1, chk2), w2);
    w3 = np.multiply(np.multiply(chk1, chk2), w3);

    return w1, w2, w3


def warp(X, Y, xControlStart, yControlStart, xControlFinal, yControlFinal, triC):
    xCB = 0
    yCB = 0
    triangles = np.array(triC.simplices.copy())  # triC.simplices.copy()
    # print(triangles)
    size = triangles.shape
    for k in range(size[0]):
        w1, w2, w3 = inTri(X, Y, xControlFinal[triangles[k, 0]], yControlFinal[triangles[k, 0]],
                           xControlFinal[triangles[k, 1]], yControlFinal[triangles[k, 1]],
                           xControlFinal[triangles[k, 2]], yControlFinal[triangles[k, 2]])
        xCB = xCB + w1 * xControlStart[triangles[k, 0]] + w2 * xControlStart[triangles[k, 1]] + w3 * xControlStart[
            triangles[k, 2]]
        yCB = yCB + w1 * yControlStart[triangles[k, 0]] + w2 * yControlStart[triangles[k, 1]] + w3 * yControlStart[
            triangles[k, 2]]
    return xCB, yCB
