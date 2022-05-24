from caliper import Caliper
import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="name of the input image")
# ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in cm)")
# args = vars(ap.parse_args())

args = {"image":'example2', "width":5}



c = Caliper(args["image"], args["width"], help = True)
print(c.image_folder, c.image_name)
image = c.img


auto = True
masked = c.hsv_filter(image, auto=auto)
image_result = c.measure(masked)
c.output()