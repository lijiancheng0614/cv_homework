import cv2
import numpy as np
from optical_flow import OpticalFlow, OpticalFlowFeatures

class VideoFeatures:
    def __init__(self, filepath):
        self.filepath = filepath
    def get_feature_vector(self):
        hists, magnitudes = self.__get_feature()
        avg_hists = np.nanmean(hists, 0)
        avg_magnitudes = np.nanmean(magnitudes, 0)
        avg_magnitudes = (avg_magnitudes - np.nanmin(avg_magnitudes)) / (
            np.nanmax(avg_magnitudes) - np.nanmin(avg_magnitudes))
        variances = np.sum(np.nanvar(hists, 0), 2)
        variances = (variances - np.nanmin(variances)) / (
            np.nanmax(variances) - np.nanmin(variances))
        feature_vector = np.concatenate((avg_hists.flatten(), avg_magnitudes.flatten(), variances.flatten()))
        return feature_vector
    def __get_feature(self):
        hists = list()
        magnitudes = list()
        x_cells, y_cells = self.__get_cells(3, 3)
        for pos, flow in enumerate(OpticalFlow(self.filepath).farneback()):
            flow_features = OpticalFlowFeatures(flow)
            hist = flow_features.get_hof(x_cells, y_cells, 8)
            magnitude = flow_features.get_magnitude(x_cells, y_cells)
            hists.append(hist)
            magnitudes.append(magnitude)
        return hists, magnitudes
    def __get_cells(self, x_guess, y_guess):
        video = cv2.VideoCapture(self.filepath)
        im = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY)
        h, w = im.shape[:2]
        return self.__closest_factor(x_guess, w), self.__closest_factor(y_guess, h)
    def __closest_factor(self, p, q):
        factors_of_q = set(reduce(list.__add__,
            ([i, q // i] for i in range(1, int(q ** 0.5) + 1) if q % i == 0)))
        return min(factors_of_q, key=lambda x : abs(x - p))
