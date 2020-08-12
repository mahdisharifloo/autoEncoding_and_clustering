#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
** DO NOT RUN THIS SCRIPY IN ANY BASE DIR THAT CONTAINS OTHER FILES
THIS SCRIPT WILL OVERWRITE ALL OF THEN IMAGE WITH MENTIONED FILE FORMAT **

This script rename all the images file name with the folloing format
.jpg, .jpeg, .png
to image hash name and the image format
<image_hash>.<image_format>


Todo:
    * should save images file name befor changeing them, in case if someone run
      this code accidently
    * chnage all images format to .jpg
    
"""
import os
from hashlib import sha256
import time

current_milli_time = lambda: int(round(time.time() * 1000))

image_base_path = '/home/amir/Python/Digi/evoke/'
acceptable_image_format = ['jpg', 'jpeg', 'png', ]
BUF_SIZE = 65536 # 64KB
os.chdir(image_base_path)
def change_all_image_name():
    changed_image_count = 0
    # tree structure of entire director
    for root, dirs, files in os.walk(image_base_path):

        # image_files = [i for i in files if i.split('.')[-1] in acceptable_image_format]
        # go throug each file
        for image in files:
            # image format to see if image should be renamed
            # and set the image foramt after renaming it
            image_format = image.split('.')[-1]
            # just select files which ends with file format in [acceptable_image_format]
            if (image_format in acceptable_image_format):
                image_dir = os.path.join(root, image)


                hasher = sha256()
                # read image to calculate it hash
                with open(image_dir, 'rb') as image_file:
                    while True:
                        data = image_file.read(BUF_SIZE)
                        if not data:
                            break
                        hasher.update(data)
                
                hash_string = hasher.hexdigest()
                # because rename add time in milli after the hash string 
                image_name_if_hased = image[:image.rfind("_")]
                # only if imagd hash never been renamed to its hash
                # don't change image name if image name is alread been changed
                if hash_string != image_name_if_hased:
                    print(f"Image File {image} hash is {hasher.hexdigest()}")
                    image_name = f"{hash_string}_{current_milli_time()}.{image_format}"
                    os.rename(image_dir, os.path.join(root, image_name))

                    changed_image_count += 1
    
    print(f"{changed_image_count} images has been renamed")

if __name__ == "__main__":
    print("are you sure, this will rename all the images in this project")
    ans = input("y/n")
    if ans.lower() == 'y':
        change_all_image_name()
    else:
        print("good, ")