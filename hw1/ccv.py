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
from itertools import product
from collections import defaultdict


class CCV(object):
    MAX_VALUE_PIXEL = 256
    NUM_BINS_EACH_COLOR = 4
    BIN_WIDTH = MAX_VALUE_PIXEL / NUM_BINS_EACH_COLOR
    TAU = 25
    def __init__(self, image_file):
        self._im_org = Image.open(image_file)
        self._w, self._h = self._im_org.size
        self._discretized_im = np.zeros((self._h, self._w), dtype=int)
        self._labeled_im = np.zeros((self._h, self._w), dtype=int)
        self._label_to_color = defaultdict(list)
        self.ccv_vector = defaultdict(list)
    def extract(self):
        self.__blur()
        self.__discretize_colorspace()
        self.__compute_connected_components()
        self.__gen_ccv_vector()
    def __blur(self):
        self._im = self._im_org.copy()
        for y in range(1, self._h - 1):
            for x in range(1, self._w - 1):
                adj_pixels = [self._im_org.getpixel((i, j))
                              for i in range(x - 1, x + 2)
                              for j in range(y - 1, y + 2)]
                self._im.putpixel((x, y), \
                    tuple(map(int, np.mean(adj_pixels, 0).tolist())))
    def __discretize_colorspace(self):
        for y, x in product(*map(range, (self._h, self._w))):
            # idx = R + G * 4 + B * 16
            idx = self.__getidx(x, y, 0) + \
                  self.__getidx(x, y, 1) + \
                  self.__getidx(x, y, 2)
            self._discretized_im[y][x] = idx
    def __getidx(self, x, y, ch = 0):
        idx = self._im.getpixel((x, y))[ch] / self.BIN_WIDTH
        return idx if ch == 0 else idx * (self.NUM_BINS_EACH_COLOR ** ch)
    def __compute_connected_components(self):
        self._current_label = 0
        for y, x in product(*map(range, (self._h, self._w))):
            checklist, xylist = self.__get_checklist(x, y)
            current_color = self._discretized_im[y][x]
            if current_color in checklist:
                # assign same label from labeled_im
                idx = checklist.index(current_color)
                cx, cy = xylist[idx][0], xylist[idx][1]
                self._labeled_im[y][x] = self._labeled_im[cy][cx]
            else:
                # assign new label
                self._labeled_im[y][x] = self._current_label
                self._label_to_color[self._current_label] = current_color
                self._current_label += 1
    def __get_checklist(self, x, y):
        checklist = []
        xylist = []
        # above left
        if x > 0 and y > 0:
            checklist.append(self._discretized_im[y - 1][x - 1])
            xylist.append([x - 1, y - 1])
        # above
        if y > 0:
            checklist.append(self._discretized_im[y - 1][x])
            xylist.append([x, y - 1])
        # above right
        if x < self._w - 1 and y > 0:
            checklist.append(self._discretized_im[y - 1][x + 1])
            xylist.append([x + 1, y - 1])
        # left
        if x > 0:
            checklist.append(self._discretized_im[y][x - 1])
            xylist.append([x - 1, y])
        return checklist, xylist
    def __gen_ccv_vector(self):
        for label in range(self._current_label):
            s = self._labeled_im[np.where(self._labeled_im == label)].size
            color = self._label_to_color[label]
            if not self.ccv_vector[color]:
                self.ccv_vector[color] = list((0, 0))
            k = 0 if s >= self.TAU else 1
            self.ccv_vector[color][k] += s


def main():
    ccv = CCV('lena.bmp')
    ccv.extract()
    for k, v in sorted(ccv.ccv_vector.items()):
        print(k, v)

if __name__ == '__main__':
    main()