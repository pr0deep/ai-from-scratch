import numpy as np
from Network import utils


class Input:
    def __init__(self, n):
        self.output = None
        self.next_layer = n

    def get_image(self, img):
        self.output = np.array(img) / 255.0

    def forward_pass(self,img):
        self.get_image(img)
        self.next_layer.forward_pass()

    def backward_pass(self,error):
        return 1

    def update(self,it,bs):
        self.next_layer.update(it,bs)

    def clear(self):
        self.output = None