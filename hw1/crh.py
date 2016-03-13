#-*- coding: utf-8 -*-

"""
Centering Refinement Histogram
1. Blur the image
   replacing each pixel value with the average value of the local neighborhood
   (including 8 adjacent pixels)
2. Discretize the colorspace into n (64 in paper) distinct colors
3. Compute the CRH
   centering: in the centermost 64% (75% in paper) of the pixels
   alpha is the number of center pixels
   beta is the number of decenter pixels

   CRH = [(alpha_1, beta_1), ..., (alpha_n, beta_n)]
"""

import os
import numpy as np
from PIL import Image


class CRH(object):
    def __init__(self, image_filepath, MAX_VALUE_PIXEL=256, NUM_BINS_EACH_COLOR=4):
        self.NUM_BINS_EACH_COLOR = NUM_BINS_EACH_COLOR
        self.BIN_WIDTH = MAX_VALUE_PIXEL / NUM_BINS_EACH_COLOR
        self.img = Image.open(image_filepath)
        self.w, self.h = self.img.size
        self.center_rectangle = ((self.w * 0.1, self.h * 0.1), \
            (self.w * 0.9, self.h * 0.9))
    def get_vector(self):
        # return crh: a list of (alpha, beta)
        crh = [list((0, 0)) for i in range(self.NUM_BINS_EACH_COLOR ** 3)]
        for x in range(self.w):
            for y in range(self.h):
                color = self.__blur(x, y)
                discretized_color = self.__discretize_colorspace(color)
                center = int(not self.__is_center(x, y))
                crh[discretized_color][center] += 1
        return crh
    def __blur(self, x, y):
        # return color: a tuple of (R, G, B)
        adj_pixels = [self.img.getpixel((i, j))
                      for i in range(x - 1, x + 2) if i >= 0 and i < self.w
                      for j in range(y - 1, y + 2) if j >= 0 and j < self.h]
        return tuple(map(int, np.mean(adj_pixels, 0).tolist()))
    def __discretize_colorspace(self, color):
        # return an int that encodes R, G, B
        return sum([color[i] / self.BIN_WIDTH \
            * (self.NUM_BINS_EACH_COLOR ** i) for i in range(3)])
    def __is_center(self, x, y):
        return x >= self.center_rectangle[0][0] and x <= self.center_rectangle[1][0] \
            and y >= self.center_rectangle[0][1] and y <= self.center_rectangle[1][1]


# if __name__ == '__main__':
#     crh = CRH('lena.bmp')
#     print(crh.get_vector())
