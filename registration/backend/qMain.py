import numpy as np
import cv2
from qNLReader import qNLReader
from qAnno import qAnno
from qmorph import qMorph
from multiprocessing import Process, Manager
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

def checkMakeDirs(dname):

    if not os.path.exists(dname):
        os.makedirs(dname)

def importNeuroLucidaXML(fname, section, OutDir, showMarkers=False, Markers='FilledCircle', Xres=1600, Yres=1200):
    
    inst = qNLReader(fname, Xres, Yres)
    inst.generateData(section, Markers)
    
    inst.drawAllContours()
    
    if(showMarkers==True):
        inst.drawAllMarkers()
        
    OutDir1 = OutDir + os.sep + 'images'
    OutDir2 = OutDir + os.sep + 'data'
    
    fname_out = os.path.basename(fname)
    imgName = os.path.splitext(fname_out)[0] + '_' + section + '.png'
    imgName = os.path.join(OutDir1, imgName)
    ptsName = os.path.splitext(fname_out)[0] + '_' + section + '_poi.csv'
    ptsName = os.path.join(OutDir2, ptsName)

    checkMakeDirs(OutDir1)
    checkMakeDirs(OutDir2)
    np.savetxt(ptsName, np.round(inst.Marker_points), delimiter=",")
    cv2.imwrite(imgName, inst.img)

def generateMasks(fname):
    img = cv2.imread(fname)
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([255, 150, 200])
    upper_blue = np.array([255, 200, 50])
    mask = cv2.inRange(img, lower_blue, upper_blue)
    
    return mask
    
def getFileNameList(dirname, ext = '*.*'):
    
    fullname = dirname + os.sep + ext
    allfiles = glob.glob(fullname)
    return allfiles



def doAnnotation(dirname, dirTemp='Templates', ind=0):

    imlist1 = getFileNameList(dirname + os.sep + 'images', '*.png')
    imlist2 = getFileNameList(dirTemp, '*.png')

    template_file = imlist2[ind]
    
    #import pdb; pdb.set_trace()
    #if __name__ == '__main__':
    for i in range(len(imlist1)):
        inst1 = qAnno("Unregistered: " + imlist1[i], imlist1[i])
        inst2 = qAnno("Template: " + template_file, template_file)
        #if __name__ == '__main__':
        p1 = Process(target=inst1.BeginAnnotation, args=())
        p2 = Process(target=inst2.BeginAnnotation, args=())

        p1.start()
        p2.start()

        p1.join()
        p2.join()


def getPOIs(fname):

    points = np.loadtxt(fname, delimiter=',')
    return points

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
                cv2.circle(img, tuple(points[i, :]), sz, col, -1)
        else:
            cv2.circle(img, tuple((points[0], points[1])), sz, col, -1)
        return img

def getMask(img):

    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([255, 150, 200])
    upper_blue = np.array([255, 200, 50])
    mask = cv2.inRange(img, lower_blue, upper_blue)

    return mask

def doBatchRegistration(inDir, warpMarkers=False, includeMarkers = False, applyMask = True, dirTemplate = 'Templates', TemplateIndex=0, Xres=1600, Yres=1200):

    imlist = getFileNameList(inDir + os.sep + 'images', '*.png')
    datlist = getFileNameList(inDir + os.sep + 'data', '*.csv')
    
    imlist2 = getFileNameList(dirTemplate, '*.png')
    template_file = imlist2[TemplateIndex]

    inDir1 = inDir + os.sep + 'wimages'
    inDir2 = inDir + os.sep + 'wdata'

    checkMakeDirs(inDir1)

    for i in range(len(imlist)):

        print(imlist[i])

        inst = qMorph(imlist[i] ,template_file)
        wimg, aimg = inst.registerImage()
        if(includeMarkers==True):
            points = getPOIs(datlist[i])
            warped_points = inst.projectCoordinates(points)
            warped_points = np.int32(np.transpose(np.squeeze(np.array(warped_points))))

            if(len(warped_points.shape) > 1):
                warped_points = warped_points[warped_points[:, 0]< Xres]
                warped_points = warped_points[warped_points[:, 1] < Yres]
            else:
                if(warped_points[0] > Xres) or (warped_points[1] > Yres):
                    warped_points[0] = 0
                    warped_points[1] = 0

            wimg = drawPoints(wimg, warped_points)

        if(warpMarkers==True):
            checkMakeDirs(inDir2)
            points = getPOIs(datlist[i])
            warped_points = inst.projectCoordinates(points)
            warped_points = np.int32(np.transpose(np.squeeze(np.array(warped_points))))
            if (len(warped_points.shape) > 1):
                warped_points = warped_points[warped_points[:, 0] < Xres]
                warped_points = warped_points[warped_points[:, 1] < Yres]
            else:
                if (warped_points[0] > Xres) or (warped_points[1] > Yres):
                    warped_points[0] = 0
                    warped_points[1] = 0
            ptsName = inDir2 + os.sep + os.path.splitext(os.path.basename(datlist[i]))[0] + '_warped.csv'
            np.savetxt(ptsName, warped_points, delimiter=",")

        if(applyMask==True):

            timg = cv2.imread(template_file)
            wmask = getMask(timg)
            #import pdb; pdb.set_trace()
            wimg = cv2.bitwise_and(wimg, wimg, mask=wmask)

        #import pdb; pdb.set_trace()
        fname = inDir1 + os.sep + os.path.splitext(os.path.basename(imlist[i]))[0] + '_warped.png'
        cv2.imwrite(fname, wimg)


def createCompositeSumImage(inDir, TemplateIndex=0, ptSize=1, singleColor=False, defaultColor = [0,0,255]):

    imlist = getFileNameList(inDir + os.sep + 'wimages', '*.png')
    datlist = getFileNameList(inDir + os.sep + 'wdata', '*.csv')

    imlist2 = getFileNameList('Templates', '*.png')
    template_file = imlist2[TemplateIndex]

    inDir1 = inDir + os.sep + 'wimages'
    inDir2 = inDir + os.sep + 'wdata'

    timg = cv2.imread(template_file)
    tmask = getMask(timg)
    fimg = np.zeros(timg.shape)

    for i in range(3):
        fimg[:,:, i] = (255-tmask)

    cnum = list(range(len(datlist)))

    if(singleColor==False):
        norm = plt.Normalize()
        colors = plt.cm.jet(norm(cnum))
        ncolors = colors.shape[0]
        colors = colors[:,0:3]
        colors = np.int32(colors*255)
        colors = np.flip(colors, axis=1)
    else:
        colors = [0,0,0]
        for i in range(len(datlist)):
            colors = np.vstack((colors, defaultColor))

        colors = np.delete(colors, 0, 0)

    for i in range(len(datlist)):

        points = np.int32(getPOIs(datlist[i]))

        # if (i == 3) or (i == 6):        #Temporary!! Please delete ASAP!
        #     points[:,0] = 1600 - points[:,0]

        drawPoints(fimg, points, colors[i,:], ptSize)

    #cv2.imshow("test", fimg)
    #cv2.waitKey(-1)

    #fimg = cv2.bitwise_and(fimg, fimg, mask=tmask)

    # sns.set()
    # ax = sns.heatmap(fimg[:,:,2])
    # plt.show(ax)
    return fimg, tmask



