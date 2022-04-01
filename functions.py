import numpy as np
import cv2



def hsv_filter(image):
    """ 
    HSV filter control panel. Return HSV filtered image.
    # """
    # Setup HSV filter trackbars
    cv2.namedWindow("TrackBars1")
    cv2.resizeWindow("TrackBars1",640,240)
    cv2.createTrackbar("Hue Min","TrackBars1",0,240, empty)
    cv2.createTrackbar("Hue Max","TrackBars1",13,240,empty)
    cv2.createTrackbar("Sat Min","TrackBars1",23,240,empty)
    cv2.createTrackbar("Sat Max","TrackBars1",239,239,empty)
    cv2.createTrackbar("Val Min","TrackBars1",0,255,empty)
    cv2.createTrackbar("Val Max","TrackBars1",77,255,empty)

    # Instructions (part 1)
    print('Instructions (part 1):')
    print('\tPress \'n\' to move onto contour detection')
    print('\tUse trackbars to adjust the HSV filter')
    
    while True:
        
        # define key press as
        k = cv2.waitKey(1)
        print(k)
        # press 'n' to break loop
        if k & 0xFF == ord('n'):
            break
        
        # convert image to HSV
        imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # get trackbar values
        h_min = cv2.getTrackbarPos("Hue Min","TrackBars1")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars1")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars1")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars1")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars1")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars1")
        
        # define HSV lower/upper bounds 
        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        
        # create and apply bitwise mask
        mask = cv2.inRange(imgHSV,lower,upper)
        masked = cv2.bitwise_and(image,image,mask=mask)
        
        # scale and display filtered image
        masked_scaled = stack_images(0.2, ([ masked ] ))
        cv2.imshow("Filter by HSV", masked_scaled)
    cv2.destroyAllWindows()
    return masked


def stack_images(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def empty(a):
    pass

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
