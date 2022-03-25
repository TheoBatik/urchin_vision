
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
# import argparse
import imutils
import cv2


# SETUP

def empty(a):
    pass

def stackImages(scale,imgArray):
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

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the input image")
# ap.add_argument("-w", "--width", type=float, required=True,
#                 help="width of the left-most object in the image (in inches)")
# args = vars(ap.parse_args())

args = {"image":'images/example2.jpg', "width":0.955}
pixelsPerMetric = None

# load the image, convert it to grayscale, and blur it slightly
img = cv2.imread(args["image"])
img_result = img.copy()

# n_dil = int(input("Enter number of erosions (input must be an integer): "))
# n_ero = int(input("Enter number of dilations (input must be an integer): "))
n_dil = 1
n_ero = 1

# HSV FILTER:

# Setup HSV filter trackbars
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,240, empty)
cv2.createTrackbar("Hue Max","TrackBars",13,240,empty)
cv2.createTrackbar("Sat Min","TrackBars",23,240,empty)
cv2.createTrackbar("Sat Max","TrackBars",239,239,empty)
cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
cv2.createTrackbar("Val Max","TrackBars",77,255,empty)

# Instructions
print('Instructions:')
print('\tPress \'q\' to exit')
print('\tUse track bars to adjust the HSV filter')

# Initialise 'contol panel' for HSV filter
while True:
    
    # define key press
    k = cv2.waitKey(1)
    
    # press 'q' to exit
    if k & 0xFF == ord('q'):
        # Save HSV values
        break
    
    # convert image to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # get trackbar values
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    
    # define HSV lower/upper bounds 
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    
    # create and apply bitwise mask
    mask = cv2.inRange(imgHSV,lower,upper)
    img_col = cv2.bitwise_and(img,img,mask=mask)
    
    # scale and display filtered image
    img_col_scaled = stackImages(0.2, ([ img_col ] ))
    cv2.imshow("Filter by HSV", img_col_scaled)

cv2.destroyAllWindows()

# MEASURMENT

# convert to grayscale and blur
kernel_blur = (7, 7)
gray = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, kernel_blur, 0)

# edges only
edged = cv2.Canny(gray, 50, 100)

# dilate and erode
kernel_de_dims = 12
kernel = np.ones((kernel_de_dims,kernel_de_dims),np.uint8)
dilated = cv2.dilate(edged, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Setup measurement trackbars
area_min_def = 4 # default
area_coeff_def = 5
area_min_max = 10 # upper bound on min area threshold
cv2.namedWindow("TrackBars2")
cv2.resizeWindow("TrackBars2",500,350)
cv2.createTrackbar("Area min","TrackBars2", area_min_def, area_min_max, empty)
cv2.createTrackbar("Area coefficient","TrackBars2", area_coeff_def, area_min_max, empty)
cv2.createTrackbar("Min Val (Canny)", "TrackBars2",50, 255, empty)
cv2.createTrackbar("Max Val (Canny)", "TrackBars2",80, 255, empty)

# total number of dilations / erosions
tot_n_dil = 0
tot_n_ero = 0

# Instructions
print('\t\'b\' to blur')
print('\t\'d\' to dilate')
print('\t\'e\' to erode')
print('\t\'f\' to fetch the contours')
print('\t\'x\' to fetch/display the bounding boxes')

# Initialise contol panel for blur, erosions, dilations and measurement
while True:
    
    try:      
        # Define key press
        k = cv2.waitKey(1)
        
        # press 'q' to exit
        if k & 0xFF == ord('q'):
            break
        
        # press 'b' to blur
        if k & 0xFF == ord('b'):
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            
        area_min_power = cv2.getTrackbarPos("Area min","TrackBars2")
        area_min_coeff = cv2.getTrackbarPos("Area coefficient","TrackBars2")
        canny_min = cv2.getTrackbarPos("Min Val (Canny)","TrackBars2")
        canny_max = cv2.getTrackbarPos("Max Val (Canny)","TrackBars2")
        
        edged = cv2.Canny(gray, canny_min, canny_max)
        
        # dilate and erode
        if k & 0xFF == ord('d'):
            dilated = cv2.dilate(edged, kernel, iterations=n_dil)
            tot_n_dil += n_dil
            print('Total number of dilations =', tot_n_dil)
            gray = dilated.copy()
            
        if k & 0xFF == ord('e'):
            eroded = cv2.erode(dilated, kernel, iterations=n_ero)
            tot_n_ero += n_ero
            print('Total number of erosions =', tot_n_ero)
            gray = eroded.copy()       
        
        if k & 0xFF == ord('f'):
            
            print('Fetching contours...')
            
            # Setup image result and containers for sufficiently large contours and bounding boxes
           
            # cnts_big = []
            # boxes = []
            
            # find contours in the edge map
            # cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Or, find contours in the dilated map
            # cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # sort the contours from left-to-right and initialize the
            cnts = contours.sort_contours(cnts, method='left-to-right')        # 'pixels per metric' calibration variable
            print('\t=> ', len(cnts[0]))
        
        
        if k & 0xFF == ord('x'):
            
            img_result = img.copy()
             
            print('Fetching bounding boxes...')
            
            # Counters for contours: in = sufficiently large; out = too small
            count_in = 0
            count_out = 0
            
            # Results container
            results = []
            
            # loop through the contours 
            for i, c in enumerate(cnts[0]):
                
                # if the contour is not sufficiently large, ignore it
                if cv2.contourArea(c) < area_min_coeff*10**area_min_power: #area_min a*10^x
                    count_out += 1
                    continue
                # print(i)
                count_in += 1
                # compute the rotated bounding box of the contour
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")
        
                # order the points in the contour such that they appear
                # on top-left, top-right, bottom-right, and bottom-left
                # order, then draw the outline of the rotated bounding box
                box = perspective.order_points(box)
                cv2.drawContours(img_result, [box.astype("int")], -1, (255, 40, 0), 2)
                cv2.drawContours(img_result, [c], 0, (0, 255, 0), 5)
                 
            
        # cv2.drawContours(img_result, boxes, -1, (0, 255, 0), q2)
            # cv2.drawContours(img_result, cnts_big, -1, (0, 255, 0), 5)
            # unpack the ordered bounding box
        #     
                (tl, tr, br, bl) = box
          
           	# compute the midpoint between the top-left aqnd top-right coordinates
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
          
              # compute the midpoint between the top-right and bottom-right
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)
                
                 	# draw the midpoints on the image
                cv2.circle(img_result, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                cv2.circle(img_result, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                cv2.circle(img_result, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                cv2.circle(img_result, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
                
                 	# draw lines between the midpoints
                cv2.line(img_result, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                		(255, 0, 255), 2)
                cv2.line(img_result, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                		(255, 0, 255), 2)
                
                 	# compute the Euclidean distance between the midpoints, in pixels
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                
                 	# if the pixels per metric has not been initialized, then
                 	# compute it as the ratio of pixels to supplied metric
                 	# (in this case, inches)
                if pixelsPerMetric is None:
                    pixelsPerMetric = dB / args["width"]
                
                 	# compute the size of the object
                dimA = dA / pixelsPerMetric
                dimB = dB / pixelsPerMetric
                
                results.append( (dimA, dimB) )
                
                 	# draw the object sizes on the image
                cv2.putText(img_result, "{:.1f}in".format(dimA),
                		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                		6, (0, 0, 0), 10)
                cv2.putText(img_result, "{:.1f}in".format(dimB),
                		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                		6, (0, 0, 0), 10)
            print('Contours: \n\t total = ', len(cnts[0]), 'Large enough = ', count_in, ' too small = ', count_out)   
            print('\t=> done.')
        
        img_result_scaled = stackImages(0.2, ([ img_result ] ))
        cv2.imshow("The measurement", img_result_scaled)
        img_stack = stackImages(0.1, ([ [img_col, edged], [dilated, eroded] ] ))
        cv2.imshow("Image stack: TL = filtered by colour; TR = edged (Canny); BL = dilated; BR = eroded ", img_stack)
        
    except Exception:
        print('Error!')
        break
    
cv2.destroyAllWindows()


# PARAMETER VALUES FINAL

print('\nFINAL PARAMETER VALUES:')
print('\thue:', h_min, h_max)
print('\tsaturation:', s_min, s_max)
print('\tvalue:', v_min, v_max)

print('\tTotal dilations:', tot_n_dil)
print('\tTotal erosions:', tot_n_ero)

print('\tblur kernel', kernel_blur)
kernel_de = (kernel_de_dims, kernel_de_dims)
print('\tDilation/erosion kernel', kernel_de)

print('\tMinimum contour area: coefficient', area_min_coeff)
print('\tMinimum contour area: power', area_min_power)
print('\tCanny value: min', canny_min)   
print('\tCanny value: max', canny_max)     

# FINAL RESULTS
print('\nRESULTS:\n', results)


# d = 1,2,3; e = 1 



# Fix the area
# Track too long to load