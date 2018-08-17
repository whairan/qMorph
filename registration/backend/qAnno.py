import cv2
import numpy as np
import pdb
import math
import pandas as pd
import os.path
import os
from sys import platform

ENTER_KEY = 10
if platform == "win32":
    ENTER_KEY = 13
elif platform == "darwin":
    ENTER_KEY = 13

SPACE_KEY = 32
BKSPACE_KEY = 8
ESC_KEY = 27

class qAnno:

    FLAG_NEWPT_CLICK = 0
    win_name = ""
    fname = ""
    CurrLabel = ''
    TempCoords = []
    img = []
    ClickList = []
    LabelList = []
    LabelList = []
    imtype = ''

    def __init__(self, wn, fn, flag="nonideal"):

        self.FLAG_NEWPT_CLICK = 0
        self.win_name = ""
        self.fname = ""
        self.CurrLabel = ''
        self.TempCoords = []
        self.img = []
        self.ClickList = []
        self.LabelList = []
        self.LabelList = []
        self.imtype = ''

        self.win_name = wn
        self.fname = fn
        self.imtype = flag

    def MouseHandler(self,event, x, y, flags, param):

        #global TempCoords, timg, CurrLabel, FLAG_NEWPT_CLICK
        #Label
        if event == cv2.EVENT_LBUTTONDOWN:
            self.FLAG_NEWPT_CLICK = 1
            self.CurrLabel = ''
            self.TempCoords = (x,y)
            self.UpdateAndDisplayLabels(self.img, DrawCurrPt=True)

        #Delete Label
        elif event == cv2.EVENT_RBUTTONDOWN:
            NewList = []
            NewLabel = []
            for i in range(len(self.ClickList)):
                dist = (x - self.ClickList[i][0])**2 + (y - self.ClickList[i][1])**2
                if(dist > 400):
                    NewList.append(self.ClickList[i])
                    NewLabel.append(self.LabelList[i])
            self.ClickList[:] = NewList
            self.LabelList[:] = NewLabel
            self.UpdateAndDisplayLabels(self.img)


    def CheckAndAppendToCoordinateList(self):

        if(len(self.TempCoords)>0):  #If a newpoint is clicked, check if it is near another point before adding
            NewList = []
            AddFlag = 1;
            for i in range(len(self.ClickList)):
                dist = (self.TempCoords[0] - self.ClickList[i][0])**2 + (self.TempCoords[1] - self.ClickList[i][1])**2
                if(dist < 25**2):
                    AddFlag = AddFlag * 0;
            if(AddFlag==1):
                self.ClickList.append(self.TempCoords)
                self.LabelList.append(self.CurrLabel)
                print(self.win_name + ": " + str(self.TempCoords))
            self.TempCoords = []

    def ModifyCoordinateListOfLabel(self,i):

        self.ClickList[i] = self.TempCoords


    def ListUpdate(self):

        #Is label non empty?
        if(len(self.CurrLabel)>0):

            DuplicateFlag=-1
            for i in range(len(self.LabelList)):
                if(self.CurrLabel==self.LabelList[i]):
                    DuplicateFlag = i;
                    break

            if(DuplicateFlag>-1):
                self.ModifyCoordinateListOfLabel(DuplicateFlag)
                return 1
            else:
                self.CheckAndAppendToCoordinateList()
                return 1

        else:
            return 0
            

    def ProcessKey(self, key):

        #global CurrLabel, LabelList, img, FLAG_NEWPT_CLICK


        if key & 0xFF == ESC_KEY:
            return -1

        if(self.FLAG_NEWPT_CLICK!=1):
            return 0

        #print(key)

        if key & 0xFF == BKSPACE_KEY:
            self.CurrLabel = self.CurrLabel[:-1]
            self.UpdateAndDisplayLabels(self.img, DrawCurrPt=True, DrawCurrLabel=True)
            #print(CurrLabel)
            return 0

        elif key & 0xFF == SPACE_KEY:
            ret = self.ListUpdate()
            if(ret==1):
                self.FLAG_NEWPT_CLICK = 0
            self.CurrLabel = ''
            return ret

        if (key & 0xFF >=48) & (key & 0xFF <=57):
            self.CurrLabel = self.CurrLabel + chr(key)
            self.UpdateAndDisplayLabels(self.img, DrawCurrPt=True, DrawCurrLabel=True)
            #print(CurrLabel)
        elif (key & 0xFF >=65) & (key & 0xFF <=90):
            self.CurrLabel = self.CurrLabel + chr(key)
            self.UpdateAndDisplayLabels(self.img, DrawCurrPt=True, DrawCurrLabel=True)
            #print(CurrLabel)
        elif (key & 0xFF >=97) & (key & 0xFF <=122):
            self.CurrLabel = self.CurrLabel + chr(key)
            self.UpdateAndDisplayLabels(self.img, DrawCurrPt=True, DrawCurrLabel=True)
            #print(CurrLabel)

    def UpdateAndDisplayLabels(self, OrgImg, DrawCurrPt=False, DrawCurrLabel=False):

        img_anno = OrgImg.copy() #np.zeros(OrgImg.shape,np.uint8)
        for i in range(len(self.ClickList)):
            cv2.circle(img_anno, self.ClickList[i], 20, (0,0,255), -1)
            cv2.putText(img_anno,self.LabelList[i],(self.ClickList[i][0]-10, self.ClickList[i][1] + 10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2, cv2.LINE_AA)

        if(DrawCurrPt == True):
            cv2.circle(img_anno, (self.TempCoords[0], self.TempCoords[1]), 20, (255, 0, 0), -1)

        if(DrawCurrLabel==True):
            cv2.putText(img_anno, self.CurrLabel, (self.TempCoords[0] - 10, self.TempCoords[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        tempimg = img_anno

        cv2.imshow(self.win_name,tempimg)
                
    def AnnotateImage(self):

        #global TempCoords, img
        cv2.namedWindow(self.win_name,cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self.MouseHandler)

        im1 = cv2.imread(self.fname, cv2.IMREAD_UNCHANGED)
        img_anno = np.zeros(im1.shape,np.uint8)

        if(self.imtype=="ideal"):
            im2 = im1[:,:,3]
            self.img = np.zeros(im1.shape,np.uint8)
            self.img[:,:,0] = im2;
            self.img[:,:,1] = im2;
            self.img[:,:,2] = im2;
        else:
            self.img = im1.copy()

        while(1):

            #CheckAndAppendToCoordinateList()
            self.UpdateAndDisplayLabels(self.img)
            k = cv2.waitKey(-1)
            ret = self.ProcessKey(k)
            if(ret==-1):
                break
            while(ret!=1):
                k = cv2.waitKey(-1)
                ret = self.ProcessKey(k)
                if(ret==-1):
                    break
            if(ret==-1):
                break

        cv2.destroyAllWindows()

    def WriteFile(self):

        #global fname, ClickList, LabelList
        fname_lab = self.fname + '.csv'
        if(len(self.ClickList)>0):               #If there is data to write...
            filew = open(fname_lab, 'w')
            for i in range(len(self.ClickList)):
                filew.write(self.LabelList[i] + ",")
                filew.write(str(self.ClickList[i][0]) + ",")
                filew.write(str(self.ClickList[i][1]) + "\n")
        else:                               #Else delete csv file if it exists
            if(os.path.isfile(fname_lab)==True):
                os.remove(fname_lab)
        
    def LoadFile(self,fname1):

        file_flag = os.path.isfile(fname1)

        if(file_flag==True):                        #If file exists
            df = pd.read_csv(fname1, header=None)
            X = list(df[1])
            Y = list(df[2])
            Labels = list(df[0])
            Coords = list(zip(X,Y))
            self.LabelList = map(str, Labels)
            self.ClickList =  map(tuple, np.int32(Coords))
            #import pdb; pdb.set_trace()
            
    def BeginAnnotation(self):

        fname_lab = self.fname + ".csv"
        self.LoadFile(fname_lab)
        self.AnnotateImage()
        self.WriteFile()
