from tkinter import *
from PIL import Image, ImageDraw
import mnist_converter
import network
import numpy as np
# Class start
# I used this class to draw 64x64 bit images

class PaintApp:

    drawing_tool = "pencil"

    # State of left mouse button
    left_button = "up"

    image_name, main_frame, drawing_area, save_button, clear_button,save_image = None, None, \
    None, None, None, None
    image = None
    draw = None
    # Positions of the mouse
    xpos, ypos = None, None

    def __init__(self, root):
        self.main_frame = Frame(root, width = 64, height = 140, bg='white')
        self.drawing_area = Canvas(self.main_frame, width = 64, height = 64)
        self.save_button = Button(self.main_frame, text="Save image")
        self.clear_button = Button(self.main_frame, text="Clear canvas")
        self.image_name = Entry(self.main_frame)
        self.save_button.bind("<ButtonPress-1>", self.save_image)
        self.clear_button.bind("<ButtonPress-1>", self.clear_canvas)
        self.drawing_area.pack()
        self.save_button.pack(side=BOTTOM, fill = X)
        self.clear_button.pack(side=BOTTOM, fill = X)
        self.image_name.pack(side=BOTTOM, fill = X)
        self.main_frame.pack()
        self.image = Image.new("RGB",(64,64),'white')
        self.draw = ImageDraw.Draw(self.image)
        # Catching events
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.left_button_down)
        self.drawing_area.bind("<ButtonRelease-1>", self.left_button_up)
        self.neur = network.Network(np.array([64*64, 4, 5,3]))
        self.mnist = mnist_converter.Converter()
        training_data = self.mnist.getMNIST()
        test_data = self.mnist.getTest()
        self.neur.SGD(training_data, 600, 6, 0.075, test_data=test_data)

    def clear_canvas(self, event):
        self.drawing_area.delete("all")
        self.image = Image.new("RGB",(64,64),'white')
        self.draw = ImageDraw.Draw(self.image)

    def save_image(self, event):
        # Name of Image
        name = self.image_name.get()
        # Saving image
        self.image.save('test'+name+'.jpg')
        feed = self.mnist.fromPath('test'+name+'.jpg')
        print 'test'+name
        output = self.neur.feedforward(feed)
        print output

    def left_button_down(self, event=None):
        self.left_button = "down"

    def left_button_up(self, event=None):
        self.left_button = "up"
        self.xpos = None
        self.ypos = None

    def motion(self, event=None):
        if self.drawing_tool == "pencil":
            self.pencil_draw(event)

    # Drawing with pencil

    def feedforward(self, event=None):
        return

    def pencil_draw(self, event=None):
        if self.left_button == "down":
            if self.xpos is not None and self.ypos is not None:
                event.widget.create_line(self.xpos, self.ypos, event.x, event.y,
                                         smooth=TRUE)
                self.draw.line([self.xpos, self.ypos, event.x, event.y],fill=(0,0,0))
            self.xpos = event.x
            self.ypos = event.y

root = Tk()
paint_app = PaintApp(root)
root.mainloop()
