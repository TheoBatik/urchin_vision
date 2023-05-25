# Measuring Sea Urchins

## Setup

Install the packages specified in requirements.txt. Any method is fine, but we recommend using pipenv. In terminal, cd into the repo and run following commands:

    install pip
    pip install pipenv
    pipenv install -r requirements.txt

## Run

With a reference object of known width in the top-left of the image, run the following command:

    pipenv run python measure.py -i image_name -f image_format -r reference_object_width

To check if it is working, run the following command and see if the output is the same as example_result.jpeg

    pipenv run python measure.py -i 'example' -f 'jpeg' -r 2.8

Alternatively, using a ratio of pixels per cm, you can run:

    pipenv run python measure.py -i image_name -f image_format -p pixels_per_cm

--i specifies the image name, 
--f is the image format/extension (e.g. 'jpg'),
--r sets the reference object width,
--p sets the pixels-to-cm ratio


# Protocol: Determining mass and diameter of live urchins 


## Set up
1.	All urchins from each experimental unit can be weighed together, thus an appropriate weight scale (ideally with at least 2 decimal places but depending on the quantity of urchins per treatment) and container must be acquired. The container must be stable on the scale and easy to tare. This could be the photo box described below.
2.	Any container could be used for the photo box provided it serves the following functions: 
    a.	Appropriate size to hold the required number of urchins, with adequate spacing between them (±1.5cm).
    b.	Somewhat standardises lighting conditions.
    c.	Standardises the distancer, retaining only the lid of the top box. Removed the bottom “floors” of the top two boxes, making a continuous box from the top lid of the upper box to the floor of the bottom box. To allow for the image to be taken, a small hole must be cut in the centre of the lid of the top box. A reference object, with a known diameter and ideally completely black, must be placed on the most left-hand side of the image, and no other objects placed further left of this object.


## Acquiring images and masse from the camera to the urchins.
A plain white background is strongly recommended. If a photo box is constructed as used in this study and described below, it will increase the likelihood that the default settings of the computer vision program will be appropriate and in general fewer alternations will be required.
3.	To replicate our photo box, attain three ‘fish’ styrofoam boxes (70cm x 35cm x18cm) and stack them on top of each oth
1.	Slowly remove the urchin basket from the water while gently shaking the basket to ensure urchins detach from the sidewall and fall into the water and remain on the bottom of the basket. Do not allow urchins to fall to the bottom of the basket once it is completely out of the water as this can damage the urchins
2.	Remove the basket from the water body and allow it to drip dry for at least 90 seconds before weighing.
3.	Remove urchins from the basket and place them into a container that has been tared and record mass.
4.	Place urchins into the photo box (or leave them in if this was the container also used to determine mass). 
5.	It is necessary to have approximately 1.5cm gaps between urchins.
6.	Ensure the reference object is on the most left-hand side of the image.
7.	Once the reference objects and urchins are placed in the bottom of the container, the lid should be closed, and a mobile phone should be placed on the lid with the camera directly above the central hole
8.	The camera should be set to a magnification of 1x without any other filters.
9.	The filename of the image must be changed to the name of the group of urchins, this name will be used to label the urchin measurement in the output CSV file.
10.	Return urchins to the body of water as soon as possible


## Processing images
1.	We found that images with a size of approximately 1.2Mb were precise while not requiring too much processing time.
2.	Input diameter of reference object for "widths" in the "args" command (line 34)
3.	Input image location
4.	Run the program, follow prompts and check if contours follow the test of the urchin	
![image](https://user-images.githubusercontent.com/102225039/174484330-cda1e7cf-947c-4361-9eaf-21ec81857c97.png)
