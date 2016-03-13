#-*- coding: utf-8 -*-

"""
1. Blur the image
   replacing each pixel value with the average value of the local neighborhood
   (including 8 adjacent pixels)
2. Discretize the colorspace into n (64 in paper) distinct colors
3. Classify the pixels as either coherent or incoherent
   - Computing connected components
     a connected component C is a maximal set of pixels with same color
     (count if two pixels are 4-connected neighbors)
   - Detemine tau's value (w * h * 0.01 in paper)
   - a pixel is coherent if the size of its C exceeds tau, otherwise incoherent
4. Compute the CCV
   alpha is the number of coherent pixels
   beta is the number of incoherent pixels

   CCV = [(alpha_1, beta_1), ..., (alpha_n, beta_n)]
"""

import os
import numpy as np
from PIL import Image


class CCV(object):
    dx = [-1, 0, 1, 0]
    dy = [0, -1, 0, 1]
    def __init__(self, image_filepath, MAX_VALUE_PIXEL=256, NUM_BINS_EACH_COLOR=4):
        self.NUM_BINS_EACH_COLOR = NUM_BINS_EACH_COLOR
        self.BIN_WIDTH = MAX_VALUE_PIXEL / NUM_BINS_EACH_COLOR
        self.img = Image.open(image_filepath)
        self.w, self.h = self.img.size
        self.tau = self.w * self.h * 0.01
    def get_vector(self):
        # return ccv: a list of (alpha, beta)
        discretized_img = np.zeros((self.w, self.h), dtype=int)
        for x in range(self.w):
            for y in range(self.h):
                color = self.__blur(x, y)
                discretized_img[x][y] = self.__discretize_colorspace(color)
        components = self.__compute_connected_components(discretized_img)
        ccv = self.__compute_ccv(components)
        return ccv
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
    def __compute_connected_components(self, discretized_img):
        # return a list of (discretized_color, count)
        visited_img = np.zeros((self.w, self.h), dtype=bool)
        components = list()
        for x in range(self.w):
            for y in range(self.h):
                if not visited_img[x][y]:
                    visited_img[x][y] = True
                    components.append([discretized_img[x][y], 1])
                    self.__floodfill(discretized_img, visited_img, components, x, y)
        return components
    def __floodfill(self, discretized_img, visited_img, components, x, y):
        # using bfs to find components
        q = [(x, y)]
        while q:
            (x, y) = q.pop(0)
            for k in range(4):
                xx = x + self.dx[k]
                yy = y + self.dy[k]
                if xx >= 0 and xx < self.w and yy >= 0 and yy < self.h and not visited_img[xx][yy] \
                    and discretized_img[xx][yy] == components[-1][0]:
                    visited_img[xx][yy] = True
                    components[-1][1] += 1
                    q.append((xx, yy))
    def __compute_ccv(self, components):
        # return ccv: a list of (alpha, beta)
        ccv = [list((0, 0)) for i in range(self.NUM_BINS_EACH_COLOR ** 3)]
        for color, count in components:
            k = 0 if count >= self.tau else 1
            ccv[color][k] += count
        return ccv


# if __name__ == '__main__':
#     ccv = CCV('lena.bmp')
#     print(ccv.get_vector())
