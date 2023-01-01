import math
import os
from PIL import Image


# Set the directory where the images are located
SOURCE_DIR = './human_images/'
TARGET_DIR = './tmp/'

# Get a list of all files in the directory
files = os.listdir(SOURCE_DIR)

# Filter the list to only include images
images = [f for f in files if f.lower().endswith('.jpg') or f.lower().endswith('.png')]

# Print the list of images
print(images)

# Loop through the images
for image in images:
    # Open the image
    im = Image.open(os.path.join(SOURCE_DIR, image))

    # Calculate the new width and height of the image
    width, height = im.size
    # aspect_ratio = width / height
    new_total_pixels = 1048576
    if width * height > new_total_pixels:
        scaling = math.sqrt((width * height) / new_total_pixels)        
    else:
        scaling = 1

    new_width = math.floor(width / scaling / 64) * 64
    new_height = math.floor(height / scaling / 64) * 64

    print('Going from (w/h):', width, height, new_width, new_height, scaling, new_height*new_height/new_total_pixels, image)

    # Resize the image
    resized_im = im.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)

    # Save the resized image
    resized_im.save(TARGET_DIR + image)