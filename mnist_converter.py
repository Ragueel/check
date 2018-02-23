import PIL.Image
import os
import numpy as np

class Converter:

    mnist_array = []
    test_data = []
    def __init__(self):
        self.getArray('./rectangles', np.array([[0], [1], [0]]))
        self.getArray('./triangles', np.array([[0], [0], [1]]))
        #self.getArray('./other', np.array([[1], [0], [0]]))
        #self.mnist_array = np.array(self.mnist_array)

    def getMNIST(self):
        return self.mnist_array

    def convert_image(self, path):
        image = PIL.Image.open(path).convert('L')
        width, height = image.size

        data = list(image.getdata())

        data = np.array([data[offset:offset + width] for offset in range(0, width * height, width)])

        array = data.ravel()
        return np.array(array)

    def getTest(self):
        root_dir = './training'
        test_dirs = os.listdir(root_dir)
        print "START"
        array_answer = np.array([[0],[1],[0]])
        for folder in test_dirs:
            for image in os.listdir(root_dir+"/"+folder):
                inputd = np.reshape(self.convert_image(open(root_dir+"/"+folder+"/"+image, 'rb')), (64*64,1))
                inputd = np.array([1.0/(x+1) for x in inputd])
                self.test_data.append([inputd
                ,array_answer])
            array_answer = np.array([[0],[0],[1]])
        return self.test_data


    def getArray(self, root_dir, image_type):
        images = os.listdir(root_dir)
        for image in images:
            inputd = np.reshape(self.convert_image(open(root_dir+"/"+image, 'rb')), (64*64,1))
            inputd = np.array([1.0/(x+1) for x in inputd])
            self.mnist_array.append([inputd
            ,image_type])

        print "Data"

converter = Converter()
