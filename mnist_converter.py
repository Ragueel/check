import PIL.Image
import os
import numpy as np

class Converter:

    def __init__(self):
        return

    def convert_image(self, path):
        image = PIL.Image.open(path).convert('L')
        width, height = image.size

        data = list(image.getdata())

        data = np.array([data[offset:offset + width] for offset in range(0, width * height, width)])

        print "Start"
        array = data.ravel()
        return array

converter = Converter()
root_dir = './rectangles'
images = os.listdir(root_dir)
filledData = np.empty([64*64, 3])
for image in images:
    print image
    np.insert(converter.convert_image(open(root_dir + image,'rb')), np.array([0,1,0]))
converter.convert_image(open('7.jpg','rb'))
