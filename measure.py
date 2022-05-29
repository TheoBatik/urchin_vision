from numpy import False_
from caliper import Caliper
import argparse

# arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="name of the input image")
ap.add_argument("-f", "--format", required=True, help="format of the input image")
ap.add_argument("-r", "--ref", type=float, required=False, help="width of the left-most object in the image (in cm)")
args = vars(ap.parse_args())

# or hard-code args
# args = {"image":'example2', "format":'jpg', "ref":5}

# set modes
auto = True
help = True
save_image = True

# create caliper 
caliper = Caliper(args["image"], args["format"], args["ref"], help=help, auto=auto)

# define input image
image = caliper.img

# apply hsv filter
masked = caliper.hsv_filter(image)

# set action sequence (if auto)
caliper.action_sequence = ['b', 'd', 'e', 'd', 'd']

# take measurement and return image result
image_result = caliper.measure(masked)

# write measurement to csv 
caliper.output(save_image=save_image)