# imports
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

    '''class for measuring the diameter of multiple urchins given input image with reference object of known length/width.'''
    
    def __init__(self, image_name, reference_object_length = None, pixels_per_cm = None, help = False):

        # calibration
        self.reference_object_length = reference_object_length
        self.pixels_per_cm = pixels_per_cm

        # load image
        self.image_name = image_name
        self.image_format = '.jpg'
        self.image_folder = 'images'
        self.path_to_image = join(self.image_folder, self.image_name + self.image_format)
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


    def hsv_filter(self, image, auto=True):

        '''Returns HSV filtered image ('masked'). 

        If auto: HSV lower & upper bounds are pulled from class attributes,
        Else: taken from trackbar values.'''

        if auto:

            hsv_lower = np.array([self.hue_min_def, self.sat_min_def, self.val_min_def])
            hsv_upper = np.array([self.hue_max_def, self.sat_max_def, self.val_max_def]) 
            masked = self.get_masked_image(image, hsv_lower, hsv_upper)
            self.update_hsv_parameters(hsv_lower, hsv_upper)

            return masked

        else:

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
        self.action_sequence.append(self.blur)
        return gray_blur

    def dilate_image(self, canny_image, total_dilations):
        dim_kernel_dilate_erode = self.parameter_values["dim_kernel_dilate_erode"]
        kernel = np.ones( dim_kernel_dilate_erode, np.uint8)
        dilation_iterations = self.parameter_values["dilation_iterations"]
        dilated = cv2.dilate(canny_image, kernel, iterations = dilation_iterations)
        self.action_sequence.append(self.dilate)
        total_dilations += dilation_iterations
        if self.help:
            print('Total number of dilations =', total_dilations)
        return dilated

    def measure(self, hsv_filtered_image):
        '''
        Initialises a control panel for the urchin diameter measurment:
        enables implemention of dilations, erosions, and blurs, contour detection,
        minimum contour area control, and fetching of minimum area bounding rectangles.
        
        Writes: measurements/parameters to .csv file

        Returns:
        image result with contours, bounding boxes and measured diameters drawn on.'''

        # convert to grayscale and blur
        gray = cv2.cvtColor(hsv_filtered_image, cv2.COLOR_BGR2GRAY)
        gray = self.blur_image(gray)

        # extract edges
        edged = cv2.Canny(gray, 50, 100)
        img_result = self.img

        # dilate and erode
        dim_kernel_dilate_erode = self.parameter_values["dim_kernel_dilate_erode"]
        kernel = np.ones( dim_kernel_dilate_erode, np.uint8)
        dilated = cv2.dilate(edged, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        self.create_measurement_trackbars()

        # total number of dilations/erosions 
        total_dilations = 0
        total_erosions = 0
        #dilation_iterations = self.parameter_values["dilation_iterations"]
        erosion_iterations = self.parameter_values["erosion_iterations"]

        # Instructions (part 2)
        if self.help:
            print('\tUse the trackbars to adjust the Canny values and minimum area')
            print('\t\'{}\' to blur'.format(self.blur))
            print('\t\'{}\' to dilate'.format(self.dilate))
            print('\t\'{}\' to erode'.format(self.erode))
            print('\t\'{}\' to fetch the contours'.format(self.get_contours))
            print('\t\'{}\' to display measurements'.format(self.take_measurement))

        while True:
            
            # try: 
                
            # get trackbar_2 values
            area_min_power = cv2.getTrackbarPos("m_area: power", self.trackbar_name_2)
            area_min_coeff = cv2.getTrackbarPos("m_area: coeff", self.trackbar_name_2)
            canny_min = cv2.getTrackbarPos("c_min", self.trackbar_name_2)
            canny_max = cv2.getTrackbarPos("c_max", self.trackbar_name_2)

            # prepare edge-map
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
                dilated = self.dilate_image(edged, total_dilations)
                gray = dilated.copy()

            # erode 
            if k & 0xFF == ord(self.erode):
                eroded = cv2.erode(dilated, kernel, iterations = erosion_iterations)
                self.action_sequence.append(self.erode)
                total_erosions += erosion_iterations
                gray = eroded.copy()
                if self.help:
                    print('Total number of erosions =', total_erosions)
            
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
                
                # setattr(self, 'img_result', self.img.copy())
                img_result = self.img

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
                    cv2.drawContours(img_result, [box.astype("int")], -1, (255, 40, 0), 2)
                    cv2.drawContours(img_result, [c], 0, (0, 255, 0), 5)
                
                    # cv2.drawContours(img_result, boxes, -1, (0, 255, 0), q2)
                    # cv2.drawContours(img_result, cnts_big, -1, (0, 255, 0), 5)
                    
                    # unpack the ordered bounding box
                    (tl, tr, br, bl) = box
                
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
                            (centreX, centreY), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 10)
                    
                    # compute the diameter, in cm
                    dimA = dA / self.pixels_per_cm
                    dimB = dB / self.pixels_per_cm
                    
                    # update results
                    self.results.append( (dimA, dimB) )
                    
                    # draw the diameters onto the image
                    cv2.putText(img_result, "{:.1f}cm".format(dimA),
                            (int(tltrX), int(tltrY)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 10)
                    cv2.putText(img_result, "{:.1f}cm".format(dimB),
                            (int(trbrX), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 10)

                # compute averages
                self.average_smaller, self.average_larger = self.tuple_average(self.results)
                self.average = self.average_overall(self.results)
                self.number_of_urchins = len(self.results) - 1
                
                if self.help:
                    print('Fetching bounding boxes...')
                    print('Contours: \n\t total = ', len(cnts[0]), 'Large enough = ', count_in, ' too small = ', count_out)   
                    print('\t=> done.')
            
            img_result_scaled = self.stack_images(0.2, ([ img_result ] ))
            setattr(self, 'img_result', img_result_scaled)
            cv2.imshow("The measurement", self.img_result)
            img_stack = self.stack_images(0.1, ([ [hsv_filtered_image, edged], [dilated, eroded] ] ))
            cv2.imshow("Image stack: TL = filtered by colour; TR = edged (Canny); BL = dilated; BR = eroded ", img_stack)
                
            # except Exception:
            #     print('Error!')
            #     break
        
        cv2.destroyAllWindows()

        # update parameter values
        self.parameter_values['Minimum contour area: coefficient'] = area_min_coeff
        self.parameter_values['Minimum contour area: power'] = area_min_power
        self.parameter_values['Canny value: min'] = canny_min
        self.parameter_values['Canny value: max'] = canny_max
        self.parameter_values['Action sequence'] = self.action_sequence

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

        ''' Returns averages based on the largest/smallest value within each tuple, given a list of tuples.'''   

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

    def output(self, save_image = True):

        '''Writes the measurements, averages, etc to a .csv file'''
        
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
            output_image_name = self.image_name + ' ' + today + ' ' + now + self.image_format
            path_to_output_image = join(cwd, self.image_folder, output_image_name)
            print(path_to_output_image)
            cv2.imwrite(path_to_output_image, self.img_result) 





