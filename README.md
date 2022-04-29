## **USAGE**

# General notes
To ensure accessibility of this machine vision program, no specialised equipment is required. 
While this does not necessarily need standardised lighting or distance from measurands, if this photo box is replicated it will be more likely the default settings of the script will work and in general less alternations will probably be required

## Instructons for aquiring images
The “photo box” can be made from three Styrofoam boxes () stacked on top of each other (example: https://drive.google.com/file/d/1zsAd7OjO3RfXeEMMIKvnMRS8KP7lDuzM/view?usp=sharing). 
The bottom “floors” of the top two boxes are cut out. 
To allow for the image to be taken, a small whole was cut in the centre of the lid of the top box. Urchins were placed at the bottom of the lowest box.
It is necessary to have approximately 1cm gaps between urchins.
A reference object with known diameter and ideally completely black must be placed on the most left-hand side of the image, no other other objects must be further left than this object.
Once the reference objects and urchins are placed in the bottom of the container, the lid should be closed, and a mobile phone should placed on the lid with the camera directly above the central hole (example: https://drive.google.com/file/d/14m4a-IbIHJP3rk5G6-rR8qfrj7mI56jq/view?usp=sharing). The camera should be set to a magnificent on 1x without any other filters etc. 
The file name of the image must be changed to the name of the group of urchins, this name will be used to label the urchin measurment in the output csv file.

# Processing images 
We found that images with a size of approximatley 1.2Mb and greater worked the best. 
Input diameter of reference object for "widths" in the "args" command (line 34) 
Input image location
Run program and check if contours follow the test of the urchin (https://drive.google.com/file/d/1E2ekjEcT-GFaTC4hx_r5qOA5mAbs5tUg/view?usp=sharing). 

# Trouble shooting:
Change exposure of image using other editting software if contour dectection is problematic
