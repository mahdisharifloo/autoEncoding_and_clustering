# -*- coding: utf-8 -*-

from PIL import Image
import resizeimage
#image = Image.open(sys.argv[1])
image_path = '/home/mahdi/Pictures/test_size.jpg'
image = Image.open(image_path)
basewidth = 500
img = image
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), Image.BICUBIC)
img.save('/home/mahdi/Pictures/sompic1.jpg')


#image = Image.open(sys.argv[1])
image_path = '/home/mahdi/Pictures/test_size.jpg'
image = Image.open(image_path)
basewidth = 500
img = image
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), Image.NEAREST)
img.save('/home/mahdi/Pictures/sompic2.jpg')