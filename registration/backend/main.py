from qAutoFit import qAutoFit
import numpy as np
from qContours import drawPoints
import cv2
import time
import glob
import os

from qContours import getUserContourFromImage, getTemplateContourFromImage, createMarkers, drawPoints
from qContourTools import getError, getFitClosest, guessAffineParameters, computeAffineMatrix, hausdorffDistance, frechetDist, findContourCenter, balloonFit

import pandas as pd
import os.path
from scipy.interpolate import splprep, splev

from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import minimize
from skimage.segmentation import active_contour
import math
import sys




	"""
	Plan:

	
	*Create an UPLOADS directory where the uploaded images will go.
	*create a textfield in html
	*Create an upload button in an html file in templates 	
	*Create a funtion in views that will correspond to the html upload button
	and will upload a selected the selected image file into the UPLOADS directory
	*Create another html button in the templates called REGISTER
	*Have the registered 
	"""