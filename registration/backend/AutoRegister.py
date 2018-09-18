from . import qAutoFit
import numpy as np
from .qContours import drawPoints
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