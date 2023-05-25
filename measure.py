from caliper import Caliper
import argparse

args = {"--image":'example', "--format":'jpeg', "--ref":5}

# arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="name of the input image")
ap.add_argument("-f", "--format", required=True, help="format of the input image")
ap.add_argument("-r", "--ref", type=float, required=False, help="width of the left-most object in the image (in cm)")
ap.add_argument("-p", "--pixel", type=float, required=False, help="ratio of pixels to cm")
args = vars(ap.parse_args())

# or hard-code the args
# args = {"image":'example', "format":'jpeg', "ref":5}

# set modes
auto = True
help = True
save_image = True

# create caliper 
caliper = Caliper(help=help, auto=auto)

# load input image
image_folder = 'images'
image_in = caliper.load_image(args["image"], args["format"], image_folder, 
    reference_object_length=args["ref"], pixels_per_cm=args["pixel"])

# apply hsv filter
masked = caliper.hsv_filter(image_in)

# set action sequence (if on auto mode)
if auto:
    caliper.action_sequence = ['d', 'e', 'e','e', 'd']

# take measurement and return image result
caliper.measure(masked)

# write measurement to csv 
caliper.output(save_image=save_image)
