
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
# import argparse
import imutils
import cv2
import pandas as pd
from os.path import exists
from datetime import date, datetime


import functions as f

# SETUP

# Functions
stack_images = f.stack_images
empty = f.empty
midpoint = f.midpoint
hsv_filter = f.hsv_filter
# big_bounding_boxes = f.big_bounding_boxes
tuple_average = f.tuple_average

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the input image")
# ap.add_argument("-w", "--width", type=float, required=True,
#                 help="width of the left-most object in the image (in inches)")
# args = vars(ap.parse_args())

args = {"image":'images/example2.jpg', "width":0.955}
pixelsPerMetric = None

# load the image and copy it
img = cv2.imread(args["image"])
img_result = img.copy()

# Parameters
dilation_iterations = 1
erosion_iterations = 1
action_sequence = []
parameter_values = {}

# HSV FILTER
# Initialise 'contol panel' for HSV filter, and return HSV-filtered / 'masked' image
#masked = hsv_filter(img)
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
    # print(k)

    # press 'n' to break loop
    if k & 0xFF == ord('n'):
        break
    
    # convert image to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
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
    masked = cv2.bitwise_and(img,img,mask=mask)
    
    # scale and display filtered image
    masked_scaled = stack_images(0.2, ([ masked ] ))
    cv2.imshow("Filter by HSV", masked_scaled)

cv2.destroyAllWindows()

# update parameter values 1
parameter_values['hue_min'] = h_min
parameter_values['hue_max'] = h_max
parameter_values['saturation_min'] = s_min
parameter_values['saturation_max'] = s_max
parameter_values['value_min'] = v_min
parameter_values['value_max'] = v_max
parameter_values['dilation_iterations'] = dilation_iterations
parameter_values['erosion_iterations'] = erosion_iterations

# MEASURMENT
## leverages dilations, erosions, blurs, contour detection and minimum area bounding boxes

# convert to grayscale and blur
results = []
kernel_blur = (7, 7)
gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, kernel_blur, 0)

# edges only
edged = cv2.Canny(gray, 50, 100)

# dilate and erode
kernel_dimension_DE = 12
kernel = np.ones((kernel_dimension_DE,kernel_dimension_DE),np.uint8)
dilated = cv2.dilate(edged, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Setup measurement trackbar
min_area_power_def = 4 # minimum area: default power
min_area_coeff_def = 5 # minimum area: default coefficient
min_area_upper_bound = 10 # greatest minimum area selectable
cv2.namedWindow("TrackBars2")
cv2.resizeWindow("TrackBars2",500,350)
cv2.createTrackbar("Min area: power","TrackBars2", min_area_power_def, min_area_upper_bound, empty)
cv2.createTrackbar("Min area: coeff","TrackBars2", min_area_coeff_def, min_area_upper_bound, empty)
cv2.createTrackbar("Canny min value", "TrackBars2",50, 255, empty)
cv2.createTrackbar("Canny max value", "TrackBars2",80, 255, empty)

# total number of dilations / erosions
total_dilations = 0
total_erosions = 0

# Operations and actions
quit = 'q'
next = 'n'
blur = 'b'
dilate = 'd'
erode = 'e'
get_contours = 'f'
get_widths = 'x'

# Instructions (part 2)
print('Instructions (part 2):')
print('\t\'b\' to blur')
print('\t\'d\' to dilate')
print('\t\'e\' to erode')
print('\t\'f\' to fetch the contours')
print('\t\'x\' to fetch/display the bounding boxes')

# Initialise contol panel for blur, erosions, dilations and measurement
while True:
    
    try: 
        
        # get trackbar2 values
        area_min_power = cv2.getTrackbarPos("Min area: power","TrackBars2")
        area_min_coeff = cv2.getTrackbarPos("Min area: coeff","TrackBars2")
        canny_min = cv2.getTrackbarPos("Canny min value","TrackBars2")
        canny_max = cv2.getTrackbarPos("Canny max value","TrackBars2")

        # Prepare edge-map
        edged = cv2.Canny(gray, canny_min, canny_max)

        # Define key press
        k = cv2.waitKey(1)
        
        # press 'q' to exit
        if k & 0xFF == ord(quit):
            break
        
        # press 'b' to blur
        if k & 0xFF == ord(blur):
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            action_sequence.append(blur)
        
        # dilate and erode
        if k & 0xFF == ord(dilate):
            dilated = cv2.dilate(edged, kernel, iterations=dilation_iterations)
            action_sequence.append(dilate)
            total_dilations += dilation_iterations
            print('Total number of dilations =', total_dilations)
            gray = dilated.copy()
            
        if k & 0xFF == ord(erode):
            eroded = cv2.erode(dilated, kernel, iterations=erosion_iterations)
            action_sequence.append(erode)
            total_erosions += erosion_iterations
            print('Total number of erosions =', total_erosions)
            gray = eroded.copy()
        
        # Get contours
        if k & 0xFF == ord(get_contours):
            
            print('Fetching contours...')
                        
            # find contours in the edge map
            # cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Or, find contours in the dilated map
            # cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            action_sequence.append(get_contours)

            # sort the contours from left-to-right and initialize the
            cnts = contours.sort_contours(cnts, method='left-to-right')        # 'pixels per metric' calibration variable
            print('\t=> ', len(cnts[0]))     
        
        # Get the bounding boxes and compute widths
        if k & 0xFF == ord(get_widths):
            
            img_result = img.copy()
             
            print('Fetching bounding boxes...')
            
            
            # Counters for contours: in = sufficiently large; out = too small
            count_in = 0
            count_out = 0           
            
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
            average_1, average_2 = tuple_average(results)
            average_overall = f.average_overall(results)
            print('Contours: \n\t total = ', len(cnts[0]), 'Large enough = ', count_in, ' too small = ', count_out)   
            print('\t=> done.')
        
        img_result_scaled = stack_images(0.2, ([ img_result ] ))
        cv2.imshow("The measurement", img_result_scaled)
        img_stack = stack_images(0.1, ([ [masked, edged], [dilated, eroded] ] ))
        cv2.imshow("Image stack: TL = filtered by colour; TR = edged (Canny); BL = dilated; BR = eroded ", img_stack)
        
    except Exception:
        print('Error!')
        break
    
cv2.destroyAllWindows()

# update parameter values 2
parameter_values['kernel_blur'] = kernel_blur
kernel_dimensions_tuple = (kernel_dimension_DE, kernel_dimension_DE)
parameter_values['Dilation/erosion kernel'] = kernel_dimensions_tuple
parameter_values['Minimum contour area: coefficient'] = area_min_coeff
parameter_values['Minimum contour area: power'] = area_min_power
parameter_values['Canny value: min'] = canny_min
parameter_values['Canny value: max'] = canny_max
parameter_values['Action sequence'] = action_sequence

# PARAMETER VALUES FINAL
print('Parameter Values:', parameter_values)

# FINAL RESULTS
print('\nRESULTS:\n', results)
print(average_1, average_2)
print('average_overall', average_overall)

# column headers
headers = ['series_name', 'number_of_urchins', 'average_overall', 'average_width_1', 'average_width_2', 'date', 'time', 'parameter_values']

# Check if csv file already exists, if not create one
file_name = 'measurements.csv'
file_exists = exists(file_name)

series_name = str( input('Enter the name of this measurement (e.g. \'Series A\'). ') )
number_of_urchins = len(results) - 1
today_unformatted = date.today()
today = today_unformatted.strftime("%d/%m/%Y")
now_unformatted = datetime.now()
now = now_unformatted.strftime("%H:%M:%S")
print('today', today)
print('now', now)

output_row = [ [series_name, number_of_urchins, average_overall, average_1, average_2, today, now, parameter_values] ]

if file_exists:
    measurements_old = pd.read_csv(file_name)
    measurements_new = pd.DataFrame(output_row, columns = headers)
    measurements = pd.concat( [measurements_old, measurements_new], ignore_index = True, axis = 0)
    measurements.to_csv(file_name, index = False)
else: 
    measurements = pd.DataFrame(output_row, columns = headers)
    measurements.to_csv(file_name, index = False)


# Fix the area
# Track too long to load