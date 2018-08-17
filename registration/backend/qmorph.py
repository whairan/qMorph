import math
import numpy as np
import cv2
import csv
from qregister import getRegisteredImage, warpCoords, LoadCSVFile, getAffineMatrix, applyAffineTransform
import os.path
import six
#   generate warped image
#   get warped coordinates
#   generate warped contours

class qMorph:

    img = []
    ref = []
    src_points = []
    dest_points = []
    src_fname = ""
    dest_fname = ""
    src_csv_fname = ""
    dest_csv_fname = ""
    FLAG_WARP_USING_IMAGES = True
    
    def __init__(self, fn1, fn2, flag=True):

        if(isinstance(fn1, six.string_types)==True):

            self.src_fname = fn1
            self.dest_fname = fn2
            self.FLAG_WARP_USING_IMAGES = flag

            if(self.FLAG_WARP_USING_IMAGES==False):

                self.src_csv_fname = self.src_fname
                self.dest_csv_fname = self.dest_fname
                
                labels, self.src_points = LoadCSVFile(self.src_csv_fname)
                labels, self.dest_points = LoadCSVFile(self.dest_csv_fname)
                

            else:
                self.src_csv_fname  = self.src_fname + ".csv"
                self.dest_csv_fname = self.dest_fname + ".csv"
                labels, self.src_points = LoadCSVFile(self.src_csv_fname)
                labels, self.dest_points = LoadCSVFile(self.dest_csv_fname)                
                self.img = np.array(cv2.imread(self.src_fname))
                self.ref = np.array(cv2.imread(self.dest_fname))
        else:
            self.FLAG_WARP_USING_IMAGES = False
            self.src_points = fn1
            self.dest_points = fn2
            
            
        self.src_points = np.array(self.src_points)
        self.dest_points = np.array(self.dest_points)
            
    
    def registerImage(self, alpha=1):
    
        if(self.FLAG_WARP_USING_IMAGES==True):

            matrix_affine, affine_img = getAffineMatrix(self.dest_points, self.src_points, self.img)
            affine_pts = applyAffineTransform(matrix_affine, self.src_points) 
            registered_img = getRegisteredImage(self.dest_points, affine_pts, affine_img, alpha)

            return registered_img, affine_img
            
    
    def projectCoordinates(self, query_pts, xres=1600, yres=1200):

        # import pdb; pdb.set_trace()
        wX, wY = warpCoords(query_pts, self.dest_points[:,0], self.dest_points[:,1], self.src_points[:,0], self.src_points[:,1], (xres, yres))
        
        return wX, wY

    











