import cv2
import numpy as np

class OpticalFlow:
    def __init__(self, filepath):
        self.filepath = filepath
    def farneback(self):
        video = cv2.VideoCapture(self.filepath)
        flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN & cv2.OPTFLOW_USE_INITIAL_FLOW
        for pos, (prev_frame, cur_frame) in enumerate(self.__iter_frames(video)):
            flow = cv2.calcOpticalFlowFarneback(prev_frame, cur_frame,
                pyr_scale=0.5, levels=3, winsize=20, iterations=5,
                poly_n=7, poly_sigma=1.5, flags=flags)
            yield flow
    def __iter_frames(self, video):
        video_length = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        prev_frame = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY)
        cur_frame = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY)
        for pos in range(video_length - 2):
            yield (prev_frame, cur_frame)
            prev_frame = cur_frame
            cur_frame = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY)

class OpticalFlowFeatures:
    def __init__(self, flow):
        self.flow = flow
    def get_hof(self, x_cells, y_cells, bins, density=True):
        hists = list()
        for x, y, _, _ in self.__iterate_cells(x_cells, y_cells):
            hist = self.__hof(x, y, bins, density)
            hists.append(hist)
        hists = np.array(hists).reshape(x_cells, y_cells, bins)
        return hists
    def get_magnitude(self, x_cells, y_cells):
        magnitude = np.zeros((x_cells, y_cells))
        for x, y, xi, yi in self.__iterate_cells(x_cells, y_cells):
            magnitude[xi, yi] = np.nanmean(np.sqrt(np.square(x) + np.square(y)))
        return magnitude
    def __iterate_cells(self, x_cells, y_cells):
        y_len, x_len, _ = self.flow.shape
        x_cl = x_len / x_cells
        y_cl = y_len / y_cells
        for i in range(x_cells):
            for j in range(y_cells):
                x_base = i * x_cl
                y_base = j * y_cl
                x_components = self.flow[y_base : y_base + y_cl, x_base : x_base + x_cl, 0]
                y_components = self.flow[y_base : y_base + y_cl, x_base : x_base + x_cl, 1]
                yield x_components, y_components, i, j
    def __hof(self, x, y, bins, density=True):
        orientations = np.arctan2(x, y)
        magnitudes = np.sqrt(np.square(x) + np.square(y))
        hist, bin_edges = np.histogram(orientations, bins=bins,
            range=(-np.pi, np.pi), weights=magnitudes, density=density)
        return hist
