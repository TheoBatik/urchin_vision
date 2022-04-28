
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

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the input image")
# ap.add_argument("-w", "--width", type=float, required=True,
#                 help="width of the left-most object in the image (in inches)")
# args = vars(ap.parse_args())

args = {"image":'images/example2.jpg', "width":5}

class Caliper():

    '''class for measuring the diameter of multiple urchins given input image with reference object of known length/width.'''
    
    def __init__(self, path_to_image, reference_object_length = None, pixels_per_cm = None, help = False):

        # calibration
        self.reference_object_length = reference_object_length
        self.pixels_per_cm = pixels_per_cm

        # load image
        self.path_to_image = path_to_image
        self.img = cv2.imread(self.path_to_image)

        # options
        self.help = help

        # parameters
        self.action_sequence = []
        self.parameter_values = {}
        self.parameter_values['dilation_iterations'] = 1
        self.parameter_values['erosion_iterations'] = 1
        self.parameter_values["dim_kernel_blur"] = (7, 7)
        self.parameter_values["dim_kernel_dilate_erode"] = (12, 12)

        # HSV trackbar default values
        self.trackbar_name_1 = "HSV Filter Trackbars"
        self.hue_min_def = 0
        self.hue_max_def = 13
        self.sat_min_def = 23
        self.sat_max_def = 239
        self.val_min_def = 0
        self.val_max_def = 77

        # setup measurement trackbar
        self.trackbar_name_2 = "Measurement Trackbars"
        self.min_area_power_def = 4 # minimum area: default power
        self.min_area_coeff_def = 5 # minimum area: default coefficient
        self.min_area_upper_bound = 10 # greatest minimum area selectable

        # action keys 
        self.actions = {'quit':'q', 'next':'n', 'blur':'b', 'dilate':'d', 'erode':'e', 'get_contours':'c', 'measure':'m'}
        self.quit = self.actions['quit']
        self.blur = self.actions['blur']
        self.dilate = self.actions['dilate']
        self.erode = self.actions['erode']
        self.get_contours = self.actions['get_contours']
        self.take_measurement = self.actions['measure']

        # results
        self.results = []
        self.img_result = self.img.copy()
        self.average = None

    # methods
    
    def average_of_smallest(self, results):
            n = len(results) - 1 # remove reference object form total urchins
            sum = 0
            for r in results:
                r0 = r[0]
                r1 = r[1]
                if r0 < r1:
                    sum += r0
                else:
                    sum += r1
            average = sum/n
            return average

    def hsv_filter(self, image):

        '''Initialise 'contol panel' for HSV filter, and return HSV-'masked' image.'''
        
        # setup HSV filter trackbars
        cv2.namedWindow(self.trackbar_name_1)
        cv2.resizeWindow(self.trackbar_name_1,640,240)
        cv2.createTrackbar("Hue Min",self.trackbar_name_1, self.hue_min_def, 180, self.empty)
        cv2.createTrackbar("Hue Max",self.trackbar_name_1, self.hue_max_def, 180,self.empty)
        cv2.createTrackbar("Sat Min",self.trackbar_name_1, self.sat_min_def ,255,self.empty)
        cv2.createTrackbar("Sat Max",self.trackbar_name_1, self.sat_max_def ,255,self.empty)
        cv2.createTrackbar("Val Min",self.trackbar_name_1, self.val_min_def ,255,self.empty)
        cv2.createTrackbar("Val Max",self.trackbar_name_1, self.val_max_def ,255,self.empty)

        # instructions (part 1)
        if self.help:
            print('Instructions (part 1):')
            print('\tPress \'n\' to move onto contour detection')
            print('\tUse trackbars to adjust the HSV filter')

        while True:
            
            # define key press as
            k = cv2.waitKey(1)

            # press 'n' to break loop
            if k & 0xFF == ord(self.quit):
                break
            
            # convert image to HSV
            imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # get trackbar values
            h_min = cv2.getTrackbarPos("Hue Min",self.trackbar_name_1)
            h_max = cv2.getTrackbarPos("Hue Max", self.trackbar_name_1)
            s_min = cv2.getTrackbarPos("Sat Min", self.trackbar_name_1)
            s_max = cv2.getTrackbarPos("Sat Max", self.trackbar_name_1)
            v_min = cv2.getTrackbarPos("Val Min", self.trackbar_name_1)
            v_max = cv2.getTrackbarPos("Val Max", self.trackbar_name_1)
            
            # define HSV lower/upper bounds 
            lower = np.array([h_min,s_min,v_min])
            upper = np.array([h_max,s_max,v_max])
            
            # create and apply bitwise mask
            mask = cv2.inRange(imgHSV,lower,upper)
            masked = cv2.bitwise_and(image,image,mask=mask)
            
            # scale and display filtered image
            masked_scaled = self.stack_images(0.2, ([ masked ] ))
            cv2.imshow("Filter by HSV", masked_scaled)

        cv2.destroyAllWindows()

        # update parameter values
        self.parameter_values['hue_min'] = h_min
        self.parameter_values['hue_max'] = h_max
        self.parameter_values['saturation_min'] = s_min
        self.parameter_values['saturation_max'] = s_max
        self.parameter_values['value_min'] = v_min
        self.parameter_values['value_max'] = v_max

        return masked

    def measure(self, hsv_filtered_image):
        '''
        Initialises control panel for urchin diameter measurment.
        Enables implemention of dilations, erosions, and blurs, contour detection,
        minimum contour area control, and fetching of minimum area bounding rectangles.
        
        Returns:
        image result with contours, bounding boxes and measured diameters drawn on.'''

        # convert to grayscale and blur
        gray = cv2.cvtColor(hsv_filtered_image, cv2.COLOR_BGR2GRAY)
        dim_kernel_blur = self.parameter_values["dim_kernel_blur"]
        gray = cv2.GaussianBlur(gray, dim_kernel_blur, 0)

        # extract edges
        edged = cv2.Canny(gray, 50, 100)

        # dilate and erode
        dim_kernel_dilate_erode = self.parameter_values["dim_kernel_dilate_erode"]
        kernel = np.ones( dim_kernel_dilate_erode, np.uint8)
        dilated = cv2.dilate(edged, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        # setup measurement trackbars
        cv2.namedWindow(self.trackbar_name_2)
        cv2.resizeWindow(self.trackbar_name_2,500,350)
        cv2.createTrackbar("Min area: power",self.trackbar_name_2, self.min_area_power_def, self.min_area_upper_bound, self.empty)
        cv2.createTrackbar("Min area: coeff",self.trackbar_name_2, self.min_area_coeff_def, self.min_area_upper_bound, self.empty)
        cv2.createTrackbar("Canny min value", self.trackbar_name_2, 50, 255, self.empty)
        cv2.createTrackbar("Canny max value", self.trackbar_name_2, 80, 255, self.empty)

        # total number of dilations/erosions 
        total_dilations = 0
        total_erosions = 0
        dilation_iterations = self.parameter_values["dilation_iterations"]
        erosion_iterations = self.parameter_values["erosion_iterations"]

        # Instructions (part 2)
        if self.help:
            print('Instructions (part 2):')
            print('\t\'b\' to blur')
            print('\t\'d\' to dilate')
            print('\t\'e\' to erode')
            print('\t\'c\' to fetch the contours')
            print('\t\'m\' to display measurements')

        while True:
            
            # try: 
                
            # get trackbar_2 values
            area_min_power = cv2.getTrackbarPos("Min area: power", self.trackbar_name_2)
            area_min_coeff = cv2.getTrackbarPos("Min area: coeff", self.trackbar_name_2)
            canny_min = cv2.getTrackbarPos("Canny min value", self.trackbar_name_2)
            canny_max = cv2.getTrackbarPos("Canny max value", self.trackbar_name_2)

            # prepare edge-map
            edged = cv2.Canny(gray, canny_min, canny_max)

            # Define key press
            k = cv2.waitKey(1)
            
            # quit loop
            if k & 0xFF == ord(self.quit):
                break
            
            # blur
            if k & 0xFF == ord(self.blur):
                gray = cv2.GaussianBlur(gray, dim_kernel_blur, 0)
                self.action_sequence.append(self.blur)
            
            # dilate
            if k & 0xFF == ord(self.dilate):
                dilated = cv2.dilate(edged, kernel, iterations = dilation_iterations)
                self.action_sequence.append(self.dilate)
                total_dilations += dilation_iterations
                print('Total number of dilations =', total_dilations)
                gray = dilated.copy()

            # erode 
            if k & 0xFF == ord(self.erode):
                eroded = cv2.erode(dilated, kernel, iterations = erosion_iterations)
                self.action_sequence.append(self.erode)
                total_erosions += erosion_iterations
                print('Total number of erosions =', total_erosions)
                gray = eroded.copy()
            
            # get contours
            if k & 0xFF == ord(self.get_contours):
                            
                # get contours from the eroded image
                cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self.action_sequence.append(self.get_contours)

                # sort the contours from left-to-right
                cnts = contours.sort_contours(cnts, method='left-to-right')
                
                if self.help:
                    print('Fetching contours...')
                    print('\t=> ', len(cnts[0]))     
            
            # fetch the bounding boxes and take measurement
            if k & 0xFF == ord(self.take_measurement):
                
                setattr(self, 'img_result', self.img.copy())
                
                # define counters for contours: _in => sufficiently large; _out => too small
                count_in = 0
                count_out = 0           
                
                # loop through the contours 
                for i, c in enumerate(cnts[0]):
                    
                    # flag reference object
                    ref_object_measured = False

                    # if the contour is not sufficiently large, ignore it
                    if cv2.contourArea(c) < area_min_coeff*10**area_min_power: #area_min a*10^x
                        count_out += 1
                        continue
                    count_in += 1

                    # compute the rotated bounding box of the contour
                    box = cv2.minAreaRect(c)
                    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                    box = np.array(box, dtype="int")
            
                    # order the points in the contour such that they appear
                    # in [top-left, top-right, bottom-right, and bottom-left]
                    # order, then draw the outline of the rotated bounding box
                    box = perspective.order_points(box)
                    cv2.drawContours(self.img_result, [box.astype("int")], -1, (255, 40, 0), 2)
                    cv2.drawContours(self.img_result, [c], 0, (0, 255, 0), 5)
                
                    # cv2.drawContours(self.img_result, boxes, -1, (0, 255, 0), q2)
                    # cv2.drawContours(self.img_result, cnts_big, -1, (0, 255, 0), 5)
                    
                    # unpack the ordered bounding box
                    (tl, tr, br, bl) = box
                
                    # compute the midpoint between the top-left aqnd top-right coordinates
                    (tltrX, tltrY) = self.midpoint(tl, tr)
                    (blbrX, blbrY) = self.midpoint(bl, br)
                
                    # compute the midpoint between the top-right and bottom-right
                    (tlblX, tlblY) = self.midpoint(tl, bl)
                    (trbrX, trbrY) = self.midpoint(tr, br)
                    
                        # draw the midpoints on the image
                    cv2.circle(self.img_result, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                    cv2.circle(self.img_result, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                    cv2.circle(self.img_result, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                    cv2.circle(self.img_result, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
                    
                    # draw lines between the midpoints
                    cv2.line(self.img_result, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
                    cv2.line(self.img_result, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
                    
                    # compute the Euclidean distance between the midpoints, in pixels
                    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                    
                        # if the pixels per metric has not been initialized, then
                        # compute it as the ratio of pixels to supplied metric
                        # (in this case, centimeters)

                    if self.pixels_per_cm is None and not ref_object_measured:
                        setattr(self, 'pixels_per_cm', dB / self.reference_object_length)
                        ref_object_measured = True
                    
                    # compute the size of the object
                    dimA = dA / self.pixels_per_cm
                    dimB = dB / self.pixels_per_cm
                    
                    # update results
                    self.results.append( (dimA, dimB) )
                    
                    # draw the urchin diameters on the image
                    cv2.putText(self.img_result, "{:.1f}cm".format(dimA),
                            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 10)
                    cv2.putText(self.img_result, "{:.1f}cm".format(dimB),
                            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 10)

                average = self.average_of_smallest(self.results)
                setattr(self, 'average', average)
                
                if self.help:
                    print('Fetching bounding boxes...')
                    print('Contours: \n\t total = ', len(cnts[0]), 'Large enough = ', count_in, ' too small = ', count_out)   
                    print('\t=> done.')
            
            # img_result_scaled = self.stack_images(0.2, ([ self.img_result ] ))
            # setattr(self, 'img_result', img_result_scaled)
            # cv2.imshow("The measurement", self.img_result)
            img_stack = self.stack_images(0.1, ([ [masked, edged], [dilated, eroded] ] ))
            cv2.imshow("Image stack: TL = filtered by colour; TR = edged (Canny); BL = dilated; BR = eroded ", img_stack)
                
            # except Exception:
            #     print('Error!')
            #     break
        
        cv2.destroyAllWindows()

        # update parameter values
        self.parameter_values['dim_kernel_blur'] = dim_kernel_blur
        self.parameter_values['Minimum contour area: coefficient'] = area_min_coeff
        self.parameter_values['Minimum contour area: power'] = area_min_power
        self.parameter_values['Canny value: min'] = canny_min
        self.parameter_values['Canny value: max'] = canny_max
        self.parameter_values['Action sequence'] = self.action_sequence

    def stack_images(self, scale, imgArray):
        '''stacks and scales a collection of image arrays. 
         https://github.com/murtazahassan/OpenCV-Python-Tutorials-and-Projects/blob/master/Basics/Joining_Multiple_Images_To_Display.py'''
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

    def empty(self, a):
        pass

    def midpoint(self, ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

        




cal = Caliper(args["image"], args["width"])

img = cal.img
masked = cal.hsv_filter(img)

cv2.imshow("The measurement", cal.img_result)
# cal.measure(masked)




# # PARAMETER VALUES FINAL
# print('Parameter Values:', parameter_values)

# # FINAL RESULTS
# print('\nRESULTS:\n', results)
# print(average_1, average_2)
# print('average_overall', average_overall)

# # column headers
# headers = ['series_name', 'number_of_urchins', 'average_overall', 'average_width_1', 'average_width_2', 'date', 'time', 'parameter_values']

# # Check if csv file already exists, if not create one
# file_name = 'measurements.csv'
# file_exists = exists(file_name)

# series_name = str( input('Enter the name of this measurement (e.g. \'Series A\'). ') )
# number_of_urchins = len(results) - 1
# today_unformatted = date.today()
# today = today_unformatted.strftime("%d/%m/%Y")
# now_unformatted = datetime.now()
# now = now_unformatted.strftime("%H:%M:%S")
# print('today', today)
# print('now', now)

# output_row = [ [series_name, number_of_urchins, average_overall, average_1, average_2, today, now, parameter_values] ]

# if file_exists:
#     measurements_old = pd.read_csv(file_name)
#     measurements_new = pd.DataFrame(output_row, columns = headers)
#     measurements = pd.concat( [measurements_old, measurements_new], ignore_index = True, axis = 0)
#     measurements.to_csv(file_name, index = False)
# else: 
#     measurements = pd.DataFrame(output_row, columns = headers)
#     measurements.to_csv(file_name, index = False)


# # Fix the area
# # Track too long to load