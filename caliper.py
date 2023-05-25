# imports
from importlib.resources import path
from pyparsing import opAssoc
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import pandas as pd
from os.path import exists, join
from os import getcwd
from datetime import date, datetime

class Caliper():
    '''
    For measuring the length & width of multiple, similar sized objects (e.g. sea urchins),
    given an image with a reference object placed in the top-left.
    
    **kwargs
    If known, pixels_per_cm can be used instead of reference_object_length.
    help=True for info to be printed in terminal.
    auto=True for no user interaction.
    '''
    
    def __init__(self, help=True, auto=False):

        # calibration
        self.reference_object_length = None
        self.pixels_per_cm = None
        # if not (bool(reference_object_length) and bool(pixels_per_cm)):
        #     print('Reference object length or pixels-to-cm ratio required.')

        # image
        self.image_name = None
        self.image_format = None
        self.image_folder = None
        self.path_to_image = None
        self.img = None

        # modes
        self.help = help
        self.auto = auto

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
        self.hue_max_def = 239
        self.sat_min_def = 0
        self.sat_max_def = 239
        self.val_min_def = 0
        self.val_max_def = 65

         # setup measurement trackbar
        self.trackbar_name_2 = "Measurement Trackbars"
        self.min_area_power_def = 2 # minimum area: default power
        self.min_area_coeff_def =8 # minimum area: default coefficient
        self.min_area_upper_bound = 10 # greatest minimum area selectable
        self.canny_min = 0
        self.canny_max = 32


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
        self.img_result = None
        self.average_smaller = None
        self.average_larger = None
        self.average = None

        # output 
        self.headers = ['basket_name' , 'date', 'time', 'number_of_urchins', 'average_of_smallest', 
            'average_of_largest', 'average_overall', 'results', 'parameter_values']
        self.file_name = 'measurements.csv'
        self.file_exists = exists(self.file_name)
        self.number_of_urchins = None

    # methods

    def load_image(self, image_name, image_format, path_to_image_folder, reference_object_length=None, pixels_per_cm=None):
        self.image_name = image_name
        self.image_format = image_format
        self.image_folder = path_to_image_folder
        path_to_image = join(path_to_image_folder, image_name + '.' + image_format)
        self.path_to_image = path_to_image
        self.reference_object_length = reference_object_length
        self.pixels_per_cm = pixels_per_cm
        image = cv2.imread(path_to_image)
        self.img = image
        if not (bool(reference_object_length) or bool(pixels_per_cm)):
            print('Warning: reference object length or pixels-to-cm ratio will be required for measurement downstream.')
        return image

    def update_hsv_parameters(self, hsv_lower, hsv_upper):

        self.parameter_values['hue_min'] = hsv_lower[0]
        self.parameter_values['hue_max'] = hsv_upper[0]
        self.parameter_values['saturation_min'] = hsv_lower[1]
        self.parameter_values['saturation_max'] = hsv_upper[1]
        self.parameter_values['value_min'] = hsv_lower[2]
        self.parameter_values['value_max'] = hsv_upper[2]

    def get_masked_image(self, image, hsv_min_array, hsv_max_array):
            
            '''Returns HSV filtered image ('masked') given arrays of the HSV lower & upper bounds, 
            and updates the HSV parameter attributes'''
            
            # convert image to HSV
            imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # create bitwise mask
            mask = cv2.inRange(imgHSV, hsv_min_array, hsv_max_array)
            # apply mask
            masked = cv2.bitwise_and(image,image,mask=mask)
            self.update_hsv_parameters(hsv_min_array, hsv_max_array)

            return masked

    def hsv_filter(self, image):

        '''Returns HSV filtered image ('masked'). 

        If auto: HSV lower & upper bounds are pulled from class attributes,
        Else: taken from trackbar values.'''
        auto = self.auto 
        if auto:

            hsv_lower = np.array([self.hue_min_def, self.sat_min_def, self.val_min_def])
            hsv_upper = np.array([self.hue_max_def, self.sat_max_def, self.val_max_def]) 
            masked = self.get_masked_image(image, hsv_lower, hsv_upper)
            self.update_hsv_parameters(hsv_lower, hsv_upper)

            return masked

        else:
            if self.help:
                print('\nManual measurement triggered...')
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
                print('\nInstructions:')
                print('\tUse the trackbars to adjust the HSV filter')
                print('\t\'{}\' to quit'.format(self.quit))

            while True:
                
                # define key press as
                k = cv2.waitKey(1)

                # press 'n' to break loop
                if k & 0xFF == ord(self.quit):
                    break
                        
                # get trackbar values
                h_min = cv2.getTrackbarPos("Hue Min",self.trackbar_name_1)
                h_max = cv2.getTrackbarPos("Hue Max", self.trackbar_name_1)
                s_min = cv2.getTrackbarPos("Sat Min", self.trackbar_name_1)
                s_max = cv2.getTrackbarPos("Sat Max", self.trackbar_name_1)
                v_min = cv2.getTrackbarPos("Val Min", self.trackbar_name_1)
                v_max = cv2.getTrackbarPos("Val Max", self.trackbar_name_1)

                # define HSV lower/upper bounds 
                hsv_lower = np.array([h_min,s_min,v_min])
                hsv_upper = np.array([h_max,s_max,v_max])
                masked = self.get_masked_image(image, hsv_lower, hsv_upper)
                
                # scale and display masked image
                masked_scaled = self.stack_images(0.2, ([ masked ] ))
                cv2.imshow("Filter by HSV", masked_scaled)

            cv2.destroyAllWindows()

            return masked

    def create_measurement_trackbars(self):
        cv2.namedWindow(self.trackbar_name_2)
        cv2.resizeWindow(self.trackbar_name_2,700,170)
        cv2.createTrackbar("m_area: power",self.trackbar_name_2, self.min_area_power_def, self.min_area_upper_bound, self.empty)
        cv2.createTrackbar("m_area: coeff",self.trackbar_name_2, self.min_area_coeff_def, self.min_area_upper_bound, self.empty)
        cv2.createTrackbar("c_min", self.trackbar_name_2, 50, 255, self.empty)
        cv2.createTrackbar("c_max", self.trackbar_name_2, 80, 255, self.empty)

    def blur_image(self, gray):
        '''Returns blured image. Input must be grayscale.'''
        dim_kernel_blur = (7, 7) #self.parameter_values["dim_kernel_blur"]
        gray_blur = cv2.GaussianBlur(gray, dim_kernel_blur, 0)
        if not self.auto:
            self.action_sequence.append(self.blur)
        return gray_blur

    def dilate_image(self, canny_image, total_dilations):
        dim_kernel_dilate_erode = self.parameter_values["dim_kernel_dilate_erode"]
        kernel = np.ones( dim_kernel_dilate_erode, np.uint8)
        dilation_iterations = self.parameter_values["dilation_iterations"]
        dilated = cv2.dilate(canny_image, kernel, iterations = dilation_iterations)
        if not self.auto:
            total_dilations += dilation_iterations
            self.action_sequence.append(self.dilate)
            if self.help:
                print('   Dilations =', total_dilations)
        return dilated, total_dilations

    def erode_image(self, dilated_image, total_erosions):
        dim_kernel_dilate_erode = self.parameter_values["dim_kernel_dilate_erode"]
        kernel = np.ones( dim_kernel_dilate_erode, np.uint8)
        erosion_iterations = self.parameter_values["erosion_iterations"]
        eroded = cv2.erode(dilated_image, kernel, iterations = erosion_iterations)
        if not self.auto:
            total_erosions += erosion_iterations
            self.action_sequence.append(self.erode)
            if self.help:
                print('   Erosions =', total_erosions)
        return eroded, total_erosions

    def get_measurement_trackbar_values(self):
        area_min_power = cv2.getTrackbarPos("m_area: power", self.trackbar_name_2)
        area_min_coeff = cv2.getTrackbarPos("m_area: coeff", self.trackbar_name_2)
        canny_min = cv2.getTrackbarPos("c_min", self.trackbar_name_2)
        canny_max = cv2.getTrackbarPos("c_max", self.trackbar_name_2)
        return area_min_power, area_min_coeff, canny_min, canny_max

    def get_contours_from_eroded_image(self, eroded_image):
        cnts, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.action_sequence.append(self.get_contours)
        # sort the contours from left-to-right
        cnts = contours.sort_contours(cnts, method='left-to-right')
        if self.help:
            length_cnts = len(cnts[0])
            print(f'   Contours fetched: {length_cnts} ')
        return cnts

    def unpack_bounding_box(self, contour, img_result):
        '''
        Computes the minimum area rectangle of input contour
        and draws it onto the image result.
        
        Returns: unpacked vertices, and image result
        '''

        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(contour)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # as [top-left, top-right, bottom-right, and bottom-left]
        box = perspective.order_points(box)

        # draw
        cv2.drawContours(img_result, [box.astype("int")], -1, (255, 40, 0), 2)
        cv2.drawContours(img_result, [contour], 0, (0, 255, 0), 5)
        
        # unpack the ordered bounding box
        (tl, tr, br, bl) = box

        return tl, tr, br, bl, img_result

    def get_midpoints_from_box(self, img_result, tl, tr, bl, br, ref_object_measured):
        '''
        Computes the midpoints given the vertices of a box;
        draws midpoints to image;
        computes the distance between opposite midpoints in pixels;
        converts this to a distance in cm using pixel_per_cm ratio.
        '''
         # compute the midpoint between the top-left aqnd top-right coordinates
        (tltrX, tltrY) = self.midpoint(tl, tr)
        (blbrX, blbrY) = self.midpoint(bl, br)
    
        # compute the midpoint between the top-right and bottom-right
        (tlblX, tlblY) = self.midpoint(tl, bl)
        (trbrX, trbrY) = self.midpoint(tr, br)
        
        # draw the midpoints on the image
        cv2.circle(img_result, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(img_result, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(img_result, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(img_result, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        
        # draw lines between the midpoints
        cv2.line(img_result, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(img_result, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
        
        # compute the Euclidean distance between the midpoints, in pixels
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # If the pixels-per-cm ratio has not been pre-calibrated/inputted, then infer it from the reference object length
        if self.pixels_per_cm is None and not ref_object_measured:
            setattr(self, 'pixels_per_cm', dB / self.reference_object_length)
            # label reference object on image
            ref_object_measured = True
            centreX = int( (trbrX + tlblX)/2 ) - 170
            centreY = int( (trbrY + tlblY)/2 )
            cv2.putText(img_result, "REF",
                (centreX, centreY), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 0, 0), 2)
        
        # compute the diameter, in cm
        dimA = dA / self.pixels_per_cm
        dimB = dB / self.pixels_per_cm
        
        # draw the diameters onto the image
        cv2.putText(img_result, "{:.1f}cm".format(dimB),
                (int(tltrX), int(tltrY)), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 0, 0), 2)
        cv2.putText(img_result, "{:.1f}cm".format(dimA),
                (int(trbrX), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 0, 0), 2)

        # update results
        self.results.append( (dimA, dimB) )

        return img_result

    def update_averages(self):
        self.average_smaller, self.average_larger = self.tuple_average(self.results)
        self.average = self.average_overall(self.results)
        self.number_of_urchins = len(self.results) - 1

    def update_measurement_parameters(self, area_min_coeff, area_min_power, canny_min, canny_max):
        self.parameter_values['Minimum contour area: coefficient'] = area_min_coeff
        self.parameter_values['Minimum contour area: power'] = area_min_power
        self.parameter_values['Canny value: min'] = canny_min
        self.parameter_values['Canny value: max'] = canny_max
        self.parameter_values['Action sequence'] = self.action_sequence

    def choose_action(self, image, action):
        '''
        Applies the morphological operation, corresponding to the given action key,
        to the input image and returns the result.
        '''
        total_dilations = 0
        total_erosions = 0
        blur = self.blur
        erode = self.erode
        dilate = self.dilate
        if action == blur:
            img_result = self.blur_image(image)
        elif action == dilate:
            img_result, _ = self.dilate_image(image, total_dilations)
        elif action == erode:
            img_result, _ = self.erode_image(image, total_erosions)
        else:
            print(f'The action given does not match {blur}, {erode}, or {dilate}. Please provide a valid action key.')
        return img_result

    def help_contour_count(self, count_in, count_out):
        print(f'\tSufficiently large => {count_in}\n\tToo small => {count_out}') 
        print('   Bounding boxes drawn.') 
        print('   Midpoints computed.')
        print('Measurment complete.')

    def check_ref_args(self):
        args_present = False
        if (self.reference_object_length or self.pixels_per_cm):
            args_present = True
        return args_present
    
    def measure(self, hsv_filtered_image):
        '''
        Implements:
        blurs, dilation, erosions, Canny edge detection, and contour extration;
        computes minimum area bounding boxes, and the length/width of the detected objects. 
        
        Mode:
        auto=False enables user to choose the implementations in response to images displayed,
        whereas auto-mode implements the morphorlogical operations above, given that the Caliper object's
        action_sequence is not None.

        Returns:
        image result with contours, bounding boxes and measured lengths/widths drawn on.

        Saves: 
        measurements and parameters as class attributes,
        '''
        
        # convert hsv_filtered_image to grayscale
        gray = cv2.cvtColor(hsv_filtered_image, cv2.COLOR_BGR2GRAY)
        
        # total number of dilations/erosions (counters)
        total_dilations = 0
        total_erosions = 0

        # mode
        auto = self.auto

        if auto:
            action_sequence = self.action_sequence
            if bool(action_sequence):
                if self.help:
                    print('\nAuto-measurement triggered...\n   Action sequence:', action_sequence)
                for action in action_sequence:
                    gray = self.choose_action(gray, action)
                
                cnts = self.get_contours_from_eroded_image(gray)
                
                # set contours counters: _in => sufficiently large; _out => too small
                count_in = 0
                count_out = 0           
                
                # set the minimum bound on contour area
                area_min_coeff = self.min_area_coeff_def
                area_min_power = self.min_area_power_def

                # define image result
                img_result = self.img

                # loop through the contours 
                for c in cnts[0]:
                    
                    # flag for reference object
                    ref_object_measured = False

                    # if the contour is not sufficiently large, ignore it
                    if cv2.contourArea(c) < area_min_coeff*10**area_min_power: # area_min = a*10^x
                        count_out += 1
                        continue
                    count_in += 1

                    tl, tr, br, bl, img_result = self.unpack_bounding_box(c, img_result)
                
                    img_result = self.get_midpoints_from_box(img_result, tl, tr, bl, br, ref_object_measured)

                setattr(self, 'img_result', img_result)
                self.update_averages()

                if self.help:
                    self.help_contour_count(count_in, count_out)

                # update measurement params
                self.update_measurement_parameters(self.min_area_coeff_def, self.min_area_power_def, self.canny_min, self.canny_max)

            else:
                if self.help:
                    print('Action sequence missing. Auto-measurement aborted.') 
        else:
            
            # Instructions (part 2)
            if self.help:
                print('\tUse the trackbars to adjust the Canny values and minimum area')
                print('\t\'{}\' to blur'.format(self.blur))
                print('\t\'{}\' to dilate'.format(self.dilate))
                print('\t\'{}\' to erode'.format(self.erode))
                print('\t\'{}\' to fetch the contours'.format(self.get_contours))
                print('\t\'{}\' to display measurements\n'.format(self.take_measurement))

            # define image result
            img_result = self.img

            # dilate and erode
            dilated = gray.copy()
            eroded = gray.copy()

            self.create_measurement_trackbars()

            while True:
                
                # try: 
                    
                # get trackbar values
                area_min_power, area_min_coeff, canny_min, canny_max = self.get_measurement_trackbar_values()

                # prepare Canny edge-map
                edged = cv2.Canny(gray, canny_min, canny_max)

                # define key press
                k = cv2.waitKey(1)
                
                # quit loop
                if k & 0xFF == ord(self.quit):
                    break
                
                # blur
                if k & 0xFF == ord(self.blur):
                    gray = self.blur_image(gray)
                
                # dilate
                if k & 0xFF == ord(self.dilate):
                    dilated, total_dilations = self.dilate_image(edged, total_dilations)
                    gray = dilated.copy()

                # erode 
                if k & 0xFF == ord(self.erode):
                    eroded, total_erosions = self.erode_image(dilated, total_erosions)
                    gray = eroded.copy()
                
                # get contours
                if k & 0xFF == ord(self.get_contours):
                    cnts = self.get_contours_from_eroded_image(eroded)   
                
                # fetch the bounding boxes and take measurement
                if k & 0xFF == ord(self.take_measurement):
                    
                    if not self.check_ref_args():
                        print('Reference_object_length or pixels_per_cm is required.')
                        break

                    img_result = self.img

                    # set contours counters: _in => sufficiently large; _out => too small
                    count_in = 0
                    count_out = 0           
                    
                    # loop through the contours 
                    for c in cnts[0]:
                        
                        # flag for reference object
                        ref_object_measured = False

                        # if the contour is not sufficiently large, ignore it
                        if cv2.contourArea(c) < area_min_coeff*10**area_min_power: #area_min a*10^x
                            count_out += 1
                            continue
                        count_in += 1

                        tl, tr, br, bl, img_result = self.unpack_bounding_box(c, img_result)
                    
                        img_result = self.get_midpoints_from_box(img_result, tl, tr, bl, br, ref_object_measured)

                    self.update_averages()
                    
                    if self.help:
                        self.help_contour_count(count_in, count_out)
                
                img_result_scaled = self.stack_images(0.2, ([ img_result ] ))
                setattr(self, 'img_result', img_result_scaled)
                cv2.imshow("The measurement", self.img_result)
                img_stack = self.stack_images(0.1, ([ [hsv_filtered_image, edged], [dilated, eroded] ] ))
                cv2.imshow("Image stack: TL = filtered by colour; TR = edged (Canny); BL = dilated; BR = eroded ", img_stack)
                    
                # except Exception:
                #     print('Error!')
                #     break
            
            cv2.destroyAllWindows()

            self.update_measurement_parameters(area_min_coeff, area_min_power, canny_min, canny_max)

            return img_result

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

    def tuple_average(self, results):
        '''
        Returns averages based on the largest/smallest value within each tuple, 
        given a list of tuples.
        '''   
        n = len(results) - 1 # remove reference object from total urchins
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
        average_of_smallest = sum_small/n
        average_of_largest = sum_big/n
        return average_of_smallest, average_of_largest

    def average_overall(self, results):

        '''Return average based on all lengths and widths combined, skipping reference object'''

        n = 2 * ( len(results) - 1 )
        sum = 0
        for i, tuple in enumerate( results ):
            # Skip reference object in calc of average
            if i == 1:
                continue
            sum += tuple[0] + tuple[1]
        average_overall = sum / n
        return average_overall

    def output(self, save_image=False):
        '''
        Writes:
        The measurement, parameters, and datetime to csv;
        and saves image result, if save_image=True. 
        '''
        # set date & time
        today_unformatted = date.today()
        today = today_unformatted.strftime("%d-%m-%Y")
        now_unformatted = datetime.now()
        now = now_unformatted.strftime("%H-%M-%S")

        # define output row
        output_row = [ [self.image_name, today, now, self.number_of_urchins, 
            self.average_smaller, self.average_larger, self.average, self.results, self.parameter_values] ]

        # if measurement file already exists insert output row, else create new file with output
        if self.file_exists:
            measurements_old = pd.read_csv(self.file_name)
            measurements_new = pd.DataFrame(output_row, columns = self.headers)
            measurements = pd.concat( [measurements_old, measurements_new], ignore_index = True, axis = 0)
            measurements.to_csv(self.file_name, index = False)
        else: 
            measurements = pd.DataFrame(output_row, columns = self.headers)
            measurements.to_csv(self.file_name, index = False)

        # save image
        if save_image:
            cwd = getcwd()
            output_image_name = self.image_name + ' ' + today + ' ' + now + '.' + self.image_format
            path_to_output_image = join(cwd, self.image_folder, output_image_name)
            cv2.imwrite(path_to_output_image, self.img_result) 




