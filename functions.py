import numpy as np
import cv2
from imutils import perspective
# from imutils import contours



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


def big_bounding_boxes(contours, image, area_min_coeff, area_min_power):
    """
    Computes the bounding box (minimum area rectangle) for each contours with sufficiently large minimum area
    and draws box onto the output image
    """
    # Counters for contours: in = sufficiently large; out = too small
    count_in = 0
    count_out = 0  
    for c in contours:
        
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < area_min_coeff*10**area_min_power: #area_min a*10^x
            count_out += 1
            continue
        count_in += 1

        # compute the bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # on top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding box
        box = perspective.order_points(box)
        cv2.drawContours(image, [box.astype("int")], -1, (255, 40, 0), 2)
        cv2.drawContours(image, [c], 0, (0, 255, 0), 5)


def tuple_average(results):
    n = len(results) - 1
    sum_small = 0
    sum_big = 0
    for i, tuple in enumerate(results):
        # Skip reference object in calc of average
        if i == 1:
            continue
        if tuple[0] < tuple[1]:
            sum_small += tuple[0]
            sum_big += tuple[1]
        else:
            sum_small += tuple[1]
            sum_big += tuple[0]
    average_1 = sum_small/n
    average_2 = sum_big/n
    return average_1, average_2

def average_overall(results):
    n = 2*(len(results) - 1)
    sum = 0
    for i, tuple in enumerate(results):
        # Skip reference object in calc of average
        if i == 1:
            continue
        sum += tuple[0] + tuple[1]
    average_overall = sum/n
    return average_overall
