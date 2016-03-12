#-*- coding: utf-8 -*-

"""
1. Blur the image
   replacing each pixel's value with the average value of the
   8 adjacent pixels
2. Discretize the colorspace into n(64 in paper) distinct colors
3. Classify pixels as either coherent or incoherent
   - Computing a connected components C for each distinct color
     C is a maximal set of pixels with same color
     (count if two pixels are eight closest neighbors)
     (C can be computed in a single pass over the image)
   - Detemine Tau's value(25 in paper)
   - C is coherent if the size of C exceeds a Tau, otherwise C is incoherent
4. Compute the CCV
   alpha is the number of coherent pixels
   beta is the number of incoherent pixels

   CCV = <(alpha_1, beta_1), ..., (alpha_n, beta_n)>
"""

import os
import numpy as np
from PIL import Image


class CCV(object):
    MAX_VALUE_PIXEL = 256
    NUM_BINS_EACH_COLOR = 4
    BIN_WIDTH = MAX_VALUE_PIXEL / NUM_BINS_EACH_COLOR
    global dx
    dx = [-1, 0, 1, 0]
    global dy
    dy = [0, -1, 0, 1]
    def __init__(self, image_file):
        self.img = Image.open(image_file)
        self.w, self.h = self.img.size
        self.tau = self.w * self.h * 0.01
    def get_vector(self):
        # return ccv_vector: a list of (alpha, beta)
        self.discretized_img = np.zeros((self.w, self.h), dtype=int)
        ccv_vector = [list((0, 0)) for i in range(self.NUM_BINS_EACH_COLOR ** 3)]
        for x in range(self.w):
            for y in range(self.h):
                # 1. blur
                color = self.img.getpixel((x, y))
                if x > 0 and x < self.w - 1 and y > 0 and y < self.h - 1:
                    adj_pixels = [self.img.getpixel((i, j))
                                  for i in range(x - 1, x + 2)
                                  for j in range(y - 1, y + 2)]
                    color = tuple(map(int, np.mean(adj_pixels, 0).tolist()))
                # 2. discretize colorspace: R + G * 4 + B * 16
                self.discretized_img[x][y] = sum([color[i] / self.BIN_WIDTH \
                    * (self.NUM_BINS_EACH_COLOR ** i) for i in range(3)])
        # 3. compute connected components, determine coherent/incoherent
        self.visited_img = np.zeros((self.w, self.h), dtype=bool)
        self.component = list() # a list of (discretized_color, count)
        for x in range(self.w):
            for y in range(self.h):
                if not self.visited_img[x][y]:
                    self.visited_img[x][y] = True
                    self.component.append([self.discretized_img[x][y], 1])
                    self.__floodfill(x, y)
        # 4. compute ccv vector
        for i in range(len(self.component)):
            color, s = self.component[i]
            k = 0 if s >= self.tau else 1
            ccv_vector[color][k] += s
        return ccv_vector
    def __floodfill(self, x, y):
        q = [(x, y)]
        while q:
            (x, y) = q.pop(0)
            for k in range(4):
                xx = x + dx[k]
                yy = y + dy[k]
                if xx >= 0 and xx < self.w and yy >= 0 and yy < self.h and not self.visited_img[xx][yy] \
                    and self.discretized_img[xx][yy] == self.component[-1][0]:
                    self.visited_img[xx][yy] = True
                    self.component[-1][1] += 1
                    q.append((xx, yy))


if __name__ == '__main__':
    # ccv = CCV('lena.bmp')
    ccv = CCV('1.jpg')
    print(ccv.get_vector())