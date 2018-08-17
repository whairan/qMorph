from xml.dom import minidom
import numpy as np
import cv2
import pdb


def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return list(reversed(tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))))

class qNLReader:

    def __init__(self, fn1, x=1600, y=1200, rsc=0.75):
        
        self.fname = ''
        self.xmldoc = []
        self.Contours = []
        self.pContours =  []
        self.Contour_colors = []
        self.Marker_points = []
        self.Params = []
        self.img = []
        self.xRes = []
        self.yRes = []
        self.rescale = []
    
        self.fname = fn1
        self.xmldoc = minidom.parse(self.fname)
        self.img = np.zeros((y, x, 3), np.uint8)
        self.xRes = x
        self.yRes = y
        self.rescale = rsc

    def getContours(self, section='S1'):

        itemlist = self.xmldoc.getElementsByTagName('contour')

        Contours = []
        ContColor = []

        for i in range(len(itemlist)):
            Cpts = itemlist[i].getElementsByTagName('point')

            if(Cpts[0].hasAttribute('sid')==True):
                sid = Cpts[0].attributes['sid'].value
            else:
                sid = 'N'

            col_val = hex_to_rgb(str(itemlist[i].attributes['color'].value))

            Coords = np.array([0,0])

            if(sid==section) or (sid=='N'):

                for j in range(len(Cpts)):

                    x =  float(Cpts[j].attributes['x'].value)
                    y =  float(Cpts[j].attributes['y'].value)

                    Coords = np.vstack(((Coords), (x, y)))

            Coords = np.delete(Coords,0,0)
            if(len(Coords.shape)>1):
                Contours.append(Coords)
                ContColor.append(col_val)

        return Contours, ContColor


    def getImageParams(self):

        resX = self.xRes
        resY = self.yRes

        All = np.concatenate(self.Contours);

        AllX = All[:,0]
        AllY = All[:,1]

        minX = np.min(AllX);
        maxX = np.max(AllX);

        minY = np.min(AllY);
        maxY = np.max(AllY);

        rangeX = maxX - minX;
        rangeY = maxY - minY;

        aspect_ratio = float(resX)/float(resY)

        reduced_res_x = self.rescale*resX
        reduced_res_y = self.rescale*resY

        scaleX = rangeX/reduced_res_x;
        scaleY = rangeY/reduced_res_y;
        scale = np.max((scaleX, scaleY))
        return minX, minY, scaleX, scaleY, scale


    def convertContoursToPixels(self):

        resX = self.xRes
        resY = self.yRes
        
        for i in range(len(self.Contours)):

            Cntr = self.Contours[i]
            X = Cntr[:, 0];
            Y = Cntr[:, 1];
            pX, pY = self.micronsToPixels(X, Y)

            pCoords = np.transpose(np.array((pX, pY)))
            self.pContours.append(pCoords)

    def micronsToPixels(self, X, Y):

        resX = self.xRes
        resY = self.yRes

        P = self.Params

        pX = np.int32((X - P[0]) / P[4])
        pY = np.int32((Y - P[1]) / P[4])

        reduced_res_x = self.rescale*resX
        reduced_res_y = self.rescale*resY

        pY = int(reduced_res_y) - pY
        pX = int((resX - reduced_res_x)/2.0) + pX
        return pX, pY

    def getMarkerCoords(self, section='S1', markerType='FilledCircle'):

        mlist = self.xmldoc.getElementsByTagName('marker')
        Coords = np.array([0, 0])

        for i in range(len(mlist)):

            mtype = str(mlist[i].attributes['type'].value)

            if(mtype==markerType) or (markerType=='All'):

                pt = mlist[i].getElementsByTagName('point')

                if (pt[0].hasAttribute('sid') == True):
                    sec = (pt[0].attributes['sid'].value)
                else:
                    sec = 'N'

                if(sec==section) or (sec=='N'):

                    for j in range(len(pt)):
                        x = float(pt[j].attributes['x'].value)
                        y = float(pt[j].attributes['y'].value)
                        Coords = np.vstack(((Coords), (x, y)))

        Coords = np.delete(Coords, 0, 0)
        if (len(Coords<2)):
            pX = 0
            pY = 0
        else:
            pX, pY = self.micronsToPixels(Coords[:,0], Coords[:,1])
        return np.transpose(np.array([pX, pY]))


    def generateData(self, sec = 'S1', mtype='FilledCircle'):
        self.Contours, self.Contour_colors = self.getContours(sec)
        self.Params = self.getImageParams()
        self.convertContoursToPixels()
        self.Marker_points = self.getMarkerCoords(sec, mtype)

    def drawAllContours(self):

        for i in range(len(self.pContours)):
            cv2.drawContours(self.img, [self.pContours[i]], -1, self.Contour_colors[i], 1)


    def drawAllMarkers(self, col=[0, 0, 255]):

        # for i in range(sz):
        #    cv2.circle(img, tuple(Coords[i, :]), 1, col, -1)
        Coords = self.Marker_points
        if (len(Coords.shape) > 1):
            self.img[Coords[:, 1], Coords[:, 0]] = col
        else:
            self.img[Coords[1], Coords[0]] = col

    def getImage(self):
        return self.img
