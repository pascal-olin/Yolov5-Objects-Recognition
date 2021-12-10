# -*- coding: utf-8 -*-
"""
Spyder Editor
POLIN may 2020
Demonstrate Contours management using numpy arrays

"""
import cv2
import numpy as np
import json
import pickle
import glob
import os
from json_tricks import dump, dumps, load, loads, strip_comments
from pathlib import Path
ix,iy = -1,-1
drawCircle = False
CONTOUR=4
#global cannyLow
#global cannyHi
cannyLow = 50
cannyHi = 200
displayContourSelectionProcess=False # show the contours selection in small frame image 
storedcontours={}
contourStoreFolder=Path("../contours/")
_im2 = None #just to test 
#print(glob.glob(contourStoreFolder+'*.npy'))

def f_removeShadows(_thisImage):
    low_H = 0
    low_S = 0 
    low_V = 0
    high_H = 182
    high_S = 35 
    high_V = 86
    _HSVImage=_thisImage
    
    if(1):
        img0 = _HSVImage
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
        
        rgb_planes = cv2.split(img)
        
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=100, beta=155, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            #norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)
        
        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)
        
        _HSVImage = result_norm
        _HSVImage = cv2.cvtColor(_HSVImage, cv2.COLOR_HSV2BGR)
        _HSVImage = cv2.cvtColor(_HSVImage, cv2.COLOR_BGR2GRAY) # we want a gray output 
        
    if(0):    
        alpha = 1 # Contrast control (1.0-3.0)
        beta = 50 # Brightness control (0-100)

        adjusted = cv2.convertScaleAbs(_HSVImage, alpha=alpha, beta=beta)
        _HSVImage = adjusted
        _HSVImage = cv2.cvtColor(_HSVImage, cv2.COLOR_BGR2GRAY) # we want a gray output 
    
    
        """ 
        unused code 
        _thisImage = cv2.cvtColor(_thisImage, cv2.COLOR_BGR2GRAY)
        #gray=_thisImage
        _thisImage=cv2.blur(_thisImage, (5,5))
        """
    if (0):
        #hsv = cv2.cvtColor(_thisImage, cv2.COLOR_BGR2HSV)
        frame_HSV = cv2.cvtColor(_HSVImage, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        mask = frame_threshold
        
        # Bitwise-AND mask and original image 
        res = cv2.bitwise_and(_HSVImage,_HSVImage, mask= mask) 
        #res=mask
        cv2.imshow('masked',res)
        _HSVImage=res
        _HSVImage=cv2.cvtColor(_HSVImage, cv2.COLOR_HSV2BGR)
        _HSVImage = cv2.cvtColor(_HSVImage, cv2.COLOR_BGR2GRAY) # we want a gray output 
        #mask = cv2.inRange(hsv, lower_red, upper_red)
        #res = cv2.bitwise_and(_thisImage,_thisImage, mask= mask)
    if (0):
        cannyLow = cv2.getTrackbarPos('cannyLow', 'Control')
        cannyHi = cv2.getTrackbarPos('cannyHi', 'Control')
        
        #gray = cv2.blur(gray, (5,5))
        #testgray = cv2.bilateralFilter(gray, 11, 17, 17) #blur. very CPU intensive.
        #cv2.imshow("Gray map", gray)
        #_im2 = gray
        #_im2=cv2.cvtColor(_thisImage, cv2.COLOR_HSV2BGR)
        #_im2 = cv2.cvtColor(_im2, cv2.COLOR_BGR2GRAY)
        #_im2 = cv2.Canny(_thisImage, cannyLow, cannyHi,apertureSize = 5,L2gradient = True)
        _HSVImage = cv2.Canny(_HSVImage, cannyLow, cannyHi)
        #_im2 = cv2.Canny(_thisImage, cannyLow, cannyHi)
    
        print("Canny values %s : %s " % (str(cannyLow),str(cannyHi)))
        
    return _HSVImage
    #return res
def f_detectContours(_thisImage,_thisThresholdType=0,_thisThresholdValue=127):
    #global _im2 # just to test 
    # proceed with contour detection from image object passed as parameter, return is a list of contours 
    _noshadow=f_removeShadows(_thisImage)
    _thval = _thisThresholdValue
    
    #cv2.imshow("test", frame)
    #trying removeshadows_im2=cv2.cvtColor(_thisImage,cv2.COLOR_BGR2GRAY)
    #im2=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #test _noshadow=cv2.medianBlur(_noshadow,5)
    #im2=cv2.bilateralFilter(im2,9,75,75)
    _h,_w,_ch = _thisImage.shape
    #_img_area=int(_h*_w)
    #ret, th0 = cv2.threshold(im2, 128, 255, 
    #                            cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    if(_thisThresholdType==0): 
        _ret,_th = cv2.threshold(_noshadow, _thval, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    elif(_thisThresholdType==1):
        _ret,_th = cv2.threshold(_noshadow,_thval,255,cv2.THRESH_BINARY)
    elif(_thisThresholdType==2):
        _th = cv2.adaptiveThreshold(_noshadow,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    elif(_thisThresholdType==3):
        _th = cv2.adaptiveThreshold(_noshadow,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    else: # default 
        _ret,_th = cv2.threshold(_noshadow, _thval, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #contours, hierarchy = cv2.findContours(th0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # find ALL contours, return as tree format 
    _contours,_hierarchy = cv2.findContours(_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    #############################################################
    return (_contours, _noshadow)


def f_createblankimage():
    
    height=480
    width=640
    blank_image = np.zeros((height,width,3), np.uint8)
    return blank_image
def nothing(n):
    pass
# build known contours dictionary 
#######################################
for filename in contourStoreFolder.glob('*.npy'):
    print (filename)
#for filename in glob.glob(contourStoreFolder+'*.npy'):
for filename in contourStoreFolder.glob('*.npy'):
    #print(glob.glob(contourStoreFolder+'*.npy'))
    #print(glob.glob(contourStoreFolder+'*.npy'))
    #with open(os.path.join(os.cwd(), filename), 'r') as f: # open in readonly mode
    # do your stuff
    #    pass
    name=str(filename).split(".npy")[0]
    shortname = name.split("\\")[-1]
    print (name)
    #name=name.split("")
    storedcontours[shortname]=np.load(filename)
    #not here ret = cv2.matchShapes(storedcontour,contoursSorted[ii],1,0.0)
    #not here print ("comparison %s with %s : result %s " % (filename,str(ii),str(ret)))
    #contour=np.load("NContour_4.txt.npy")
    #blank_image = cv2.drawContours(blank_image, storedcontour, -1, (255, 127, 0), 2)
mythreshold = 66   
mythfunction = 10
#global cannyLow 
#global cannyHi
cv2.namedWindow('Control')
cv2.createTrackbar("mythreshold", "Control", mythreshold, 255, nothing)
cv2.createTrackbar("mythfunction", "Control", mythfunction, 255, nothing)
cv2.createTrackbar("cannyLow", "Control", cannyLow, 255, nothing)
cv2.createTrackbar("cannyHi", "Control", cannyHi, 255, nothing)

# mouse callback function
def f_mouseCallback(event,x,y,flags,param):
    global ix,iy
    global image
    global drawCircle
    global contours
    global contoursSorted # sorted by size contours 
    if event == cv2.EVENT_RBUTTONDBLCLK:
        # np.set_printoptions(threshold=np.inf)
        # coords = np.array2string(contours[CONTOUR])
        # open("contour_%d.txt" % CONTOUR, "w").write(coords)
        uu = len(contoursSorted)
        #print("selecting contour")
        for i in range(0,uu):
            #print(i)
            #r=cv2.pointPolygonTest(contoursSorted[i],Point(y,x),False)
            # check which contour mouse pointer (x,y) is in (may be several if tree selection is used)
            r=cv2.pointPolygonTest(contoursSorted[i],(x,y),False)
            if r>0:
                print("Selected contour "+str(i)+" "+str(contourStoreFolder))
                #np.set_printoptions(threshold=np.inf)
                # not reversible coords = np.array2string(contoursSorted[i])
                # not reversible reversibleCoords=pickle.dumps(contoursSorted[i], protocol=0) # protocol 0 is printable ASCII
                # not reversible open(contourStoreFolder+"contour_%d.txt" % i, "w").write(coords)
                #write contour's matrix on disk using np.save for reversibility 
                np.save(str(contourStoreFolder)+"\\NContour_%d.txt" % i,contoursSorted[i])
                # to keep _str = pickle.dumps(contoursSorted[i])
                # to keep open(contourStoreFolder+"Picklecontour_%d." % i, "wb").write(_str)
            
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if (drawCircle):
            drawCircle = False
        else: drawCircle=True

        print("mouse event called " + str(x)+str(y))
        #cv2.circle(frame,(x,y),100,(255,255,0),-1)
        ix,iy = x,y

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
cv2.namedWindow("processed")
cv2.setMouseCallback("test",f_mouseCallback)
cv2.setMouseCallback("processed",f_mouseCallback)
img_counter = 0
"""
# image colors thresholds. 
Red [160,170,50] [179,250,220]
Green [53,74,160] [90,147,255]
Blue [110,50,150] [130,260,255]
Yellow [20,100,100] [30,255,255]

"""
######## Main loop #################
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    #cv2.namedWindow("test")
    _h,_w,_ch = frame.shape
    _img_area=int(_h*_w)
    _th = cv2.getTrackbarPos('mythreshold', 'Control')
    _func = cv2.getTrackbarPos('mythfunction', 'Control')
    #global cannyLow 
    #global cannyHi

    if (_func < 50):
        _myfunc=0
    elif(_func < 100):
        _myfunc=1
    elif(_func < 150):
        _myfunc=2
    elif(_func < 200):
        _myfunc=3     
    # # proceed with contour detection from image object passed as parameter, return is a list of contours 
    contours,noshadow = f_detectContours(frame,_myfunc,_th)
    #print("Number of Contours found = " + str(len(contours)))
    # sort contours by size 
    contoursSorted = sorted(contours, key=cv2.contourArea,reverse=True)
    # another sorting method : n=len(contours)-1    
    #Sort the contours by area and then remove the largest frame contour
    #contours=sorted(contours,key=cv2.contourArea,reverse=False)[:n]
    image = frame.copy()
    #im2=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #im2=cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  #add this line
    #deleteme im2=frame.copy()
    #image = cv2.drawContours(im2, contoursSorted[1:65], -1, (255, 0, 0), 1)
    # blank image to show contour selection process 
    blank_image = f_createblankimage()
    u = int(len(contoursSorted)-1)
    if (u>2):
        #for ii in range(2,u):
        for ii in range(1,u):
            if (int(cv2.contourArea(contoursSorted[ii])) >_img_area/1000 and int(cv2.contourArea(contoursSorted[ii])) <_img_area/25):
                _a = cv2.moments(contoursSorted[ii])
                
                image = cv2.drawContours(image, contoursSorted[ii], -1, (255, 0, 0), 2)
                #print(str(_a))
                #print(str(int(_a['m10']/_a['m00'])))
                #print("nothing")
                if (_a['m00']):
                    cx=(int(_a['m10']/_a['m00']))
                    cy=(int(_a['m01']/_a['m00']))
                    image = cv2.putText(image,str(ii),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
                    # try our loaded contours 
                # default value used to keep only best contour     
                _retBest = 1
                _objBest= "dummy"
                _contoursSortederBest={}
                _contoursSortederBest['m00'] = 1
                #######################################
                for _objname in storedcontours:
                #######################################
                    # we shift the contour left and up as much as we can
                    # note that this should have been done at contour record event
                    height=180
                    width=240
                    # translate contour top left, calculate centroid 
                    dM=cv2.moments(storedcontours[_objname])
                    # calculate new centroid relative to this small frame
                    dcx = int((dM['m10']/dM['m00'])-width/2)
                    dcy = int((dM['m01']/dM['m00'])-height/2)
                    # move contour left and up by calculated amount 
                    storedcontours[_objname]=storedcontours[_objname]-[dcx,dcy]
                    if (displayContourSelectionProcess):
                        blank_image = np.zeros((height,width,3), np.uint8)
                        cv2.imshow("objects", blank_image)
                        k = cv2.waitKey(1)
                        blank_image = cv2.drawContours(blank_image, storedcontours[_objname],-1, (0,0,255),3)
                        cv2.imshow("objects", blank_image) 
                        k = cv2.waitKey(2)
                        if k%256 == 27:
                            # ESC pressed
                            print("Escape hit, closing...")
                            break
                    ret = cv2.matchShapes(storedcontours[_objname],contoursSorted[ii],1,0.0)
                    _contoursSorteder = cv2.moments(contoursSorted[ii])
                    _blue=255
                    _red=255
                    # if the match is better then make sure the best object (name, match value and centroid) are returned
                    if ret < _retBest:
                        _objBest = _objname
                        _retBest = ret 
                        _contoursSortederBest = _contoursSorteder 
                _contoursSorteder=_contoursSortederBest    
                if (_a['m00'] and _retBest <= 0.2 and not ( "doh" in _objBest)):
                    cx=(int(_a['m10']/_a['m00']))
                    cy=(int(_a['m01']/_a['m00']))
                    name=_objBest+str("(%.2f)" % round(_retBest,2))
                    _blue=int((255-(255*(_retBest/0.2))))
                    _red=int(255*(_retBest/0.2))
                    frame = cv2.putText(frame,name,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(_blue,17,_red),2)
                #######################################
    
    if(drawCircle):
        cv2.circle(image,(ix,iy),100,(255,255,0),1)
    cv2.imshow("processed",image)	
    cv2.imshow("original", frame) 
    cv2.imshow("noshadow", noshadow)            
    k = cv2.waitKey(2000)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressedm take picture !
        pass
        # attention this is valid for linux paths only (to be tested on windows) 
        path = './images'
        # path = "/home/po/Documents/2021-10-objects-recognition-CNN-Yolov5"

        img_name = "desk_{}.png".format(img_counter)
        cv2.imwrite(os.path.join(path , img_name),frame)
        # cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
