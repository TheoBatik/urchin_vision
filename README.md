# Measuring Sea Urchins

## Setup

Install the packages specified in requirements.txt. Any method is fine, but we recommend using pipenv. In terminal, cd into the repo and run following commands:

    install pip
    pip install pipenv
    pipenv install -r requirements.txt

## Run

With a reference object of known width in the top-left of the image, run the following command:

    pipenv run python measure.py -i image_name -f image_format -w width

Alternatively, using a ratio of pixels per cm, you can run:

    pipenv run python measure.py -i image_name -f image_format -p pixels_per_cm

--i specifies the image name, 
--f is the image format/extension (e.g. 'jpg'),
--w sets the reference object width,
--p sets the pixels-to-cm ratio

For example, you could run
    
    pipenv run python measure.py -i example2 -f jpg -w 5

## General notes

To ensure accessibility no specialised equipment is required. 

While lighting or distance from measurands does not need to be standardised, if this photo box described is replicated it will increase likelihood that the default settings of the script will be appropriate and in general less alternations will be required

## Instructions for acquiring images

The “photo box” can be made from three common styrofoam boxes (70cm x 35cm x18cm) stacked on top of each other (example: https://drive.google.com/file/d/1zsAd7OjO3RfXeEMMIKvnMRS8KP7lDuzM/view?usp=sharing).

The bottom “floors” of the top two boxes are cut out. 

To allow for the image to be taken, a small hole must be cut in the centre of the lid of the top box. 

A reference object, with known diameter and ideally completely black, must be placed on the most left-hand side of the image, no other objects must be further left than this object.

Once the reference objects and urchins are placed in the bottom of the container, the lid should be closed, and a mobile phone should be placed on the lid with the camera directly above the central hole (example: https://drive.google.com/file/d/14m4a-IbIHJP3rk5G6-rR8qfrj7mI56jq/view?usp=sharing). 

The camera should be set to a magnificent on 1x without any other filters etc. 

It is necessary to have approximately 1cm gaps between urchins.

The file name of the image must be changed to the name of the group of urchins, this name will be used to label the urchin measurement in the output csv file, and in the image result (if saved).

## Processing images 

We found that images with a size of approximately 1.2Mb and greater were the most accurate. 

Input diameter of reference object for "widths" in the "args" command (line 34) 

Input image location

Run program, follow prompts and check if contours follow the test of the urchin (https://drive.google.com/file/d/1E2ekjEcT-GFaTC4hx_r5qOA5mAbs5tUg/view?usp=sharing). 

## Trouble shooting:

If edge detection is not accurate enough, generally only the val (Value) of the HSV filter needs to be adjusted.

Increase exposure of image using other editing software if contour detection is not working with editting tools within script. 


