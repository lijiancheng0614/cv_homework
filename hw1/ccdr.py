#-*- coding: utf-8 -*-

"""
Color Coherence Distance Refinement
1. Blur the image
   replacing each pixel value with the average value of the local neighborhood
   (including 8 adjacent pixels)
2. Discretize the colorspace into n (64 in paper) distinct colors
3. Classify the pixels as either coherent or incoherent, center or decenter
   - Computing connected components
     a connected component C is a maximal set of pixels with same color
     (count if two pixels are 4-connected neighbors)
   - Detemine tau's value (w * h * 0.01 in paper)
   - a pixel is coherent if the size of its C exceeds tau, otherwise incoherent
   - a pixel is center if in the centermost 64% (75% in paper) of the pixels, otherwise decenter
4. Compute the CCDR
   a is the number of coherent, center pixels
   b is the number of incoherent, center pixels
   c is the number of coherent, decenter pixels
   d is the number of incoherent, decenter pixels

   CCDR = [[a_1, b_1, c_1, d_1], ..., [a_n, b_n, c_n, d_n]]
"""

import os
import numpy as np
from PIL import Image


class CCDR(object):
    dx = [-1, 0, 1, 0]
    dy = [0, -1, 0, 1]
    def __init__(self, image_filepath, MAX_VALUE_PIXEL=256, NUM_BINS_EACH_COLOR=4):
        self.NUM_BINS_EACH_COLOR = NUM_BINS_EACH_COLOR
        self.BIN_WIDTH = MAX_VALUE_PIXEL / NUM_BINS_EACH_COLOR
        self.img = Image.open(image_filepath)
        self.w, self.h = self.img.size
        self.center_rectangle = ((int(self.w * 0.1), int(self.h * 0.1)), \
            (int(self.w * 0.9), int(self.h * 0.9)))
        self.tau = self.w * self.h * 0.01
    def get_vector(self):
        # return ccdr: a list of (alpha, beta)
        discretized_img = np.zeros((self.w, self.h), dtype=int)
        for x in range(self.w):
            for y in range(self.h):
                color = self.__blur(x, y)
                discretized_img[x][y] = self.__discretize_colorspace(color)
        components = self.__compute_connected_components(discretized_img)
        ccdr = self.__compute_ccdr(components)
        return ccdr
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
        components = [list(), list()]
        # decenter
        for x in range(self.center_rectangle[0][0], self.center_rectangle[1][0] + 1):
            for y in range(self.center_rectangle[0][1], self.center_rectangle[1][1] + 1):
                visited_img[x][y] = True
        for x in range(self.w):
            for y in range(self.h):
                if not visited_img[x][y]:
                    visited_img[x][y] = True
                    components[1].append([discretized_img[x][y], 1])
                    self.__floodfill(discretized_img, visited_img, components[1], x, y)
        # center
        for x in range(self.center_rectangle[0][0], self.center_rectangle[1][0] + 1):
            for y in range(self.center_rectangle[0][1], self.center_rectangle[1][1] + 1):
                visited_img[x][y] = False
        for x in range(self.w):
            for y in range(self.h):
                if not visited_img[x][y]:
                    visited_img[x][y] = True
                    components[0].append([discretized_img[x][y], 1])
                    self.__floodfill(discretized_img, visited_img, components[0], x, y)
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
    def __compute_ccdr(self, components):
        # return ccdr: a list of (alpha, beta)
        ccdr = [[0 for j in range(4)] for i in range(self.NUM_BINS_EACH_COLOR ** 3)]
        for i in range(2):
            for color, count in components[i]:
                k = i * 2 if count >= self.tau else i * 2 + 1
                ccdr[color][k] += count
        return ccdr


# if __name__ == '__main__':
#     ccdr = CCDR('lena.bmp')
#     print(ccdr.get_vector())
