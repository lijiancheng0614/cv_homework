import cv2
import numpy as np

class Flow:
    def __init__(self, vectors, cur_frame, prev_frame, cur_frame_color):
        self.vectors = vectors
        self.cur_frame = cur_frame
        self.prev_frame = prev_frame
        self.cur_frame_color = cur_frame_color
    @staticmethod
    def draw_flow(frame, im, vectors, step=16):
        mult = 4
        h, w = im.shape[:2]
        y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1)
        fx, fy = vectors[y, x].T
        # create line endpoints
        lines = np.vstack([x, y, x + (fx * mult), y + (fy * mult)]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        for pos, ((x1, y1), (x2, y2)) in enumerate(lines):
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 1, cv2.CV_AA)
            cv2.circle(frame, (x1, y1), 1, (255, 255, 0), 1, cv2.CV_AA)
        return frame
    def show(self, title="Optical Flow", display_flow=True, text=None, display=True):
        if title == "Prediction":
            frame = self.cur_frame_color.copy()
        else:
            frame = self.cur_frame
        if display_flow:
            self.draw_flow(frame, self.cur_frame, self.vectors)
        if text:
            h, w = frame.shape[:2]
            cv2.putText(frame, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        if display:
            cv2.imshow(title, frame)
            if cv2.waitKey(30) & 0xff == 27:
                exit()
        return frame

class OpticalFlow:
    def __init__(self, filepath):
        self.filepath = filepath
    def farneback(self):
        video = cv2.VideoCapture(self.filepath)
        flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN & cv2.OPTFLOW_USE_INITIAL_FLOW
        for pos, (prev_frame, cur_frame, cur_frame_color) in enumerate(self._iter_frames(video)):
            flow = cv2.calcOpticalFlowFarneback(prev_frame, cur_frame,
                pyr_scale=0.5, levels=3, winsize=20, iterations=5,
                poly_n=7, poly_sigma=1.5, flags=flags)
            yield Flow(flow, cur_frame, prev_frame, cur_frame_color)
    def _iter_frames(self, video):
        video_length = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        prev_frame = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY)
        cur_frame_color = video.read()[1]
        cur_frame = cv2.cvtColor(cur_frame_color, cv2.COLOR_BGR2GRAY)
        for pos in range(video_length - 2):
            yield (prev_frame, cur_frame, cur_frame_color)
            prev_frame = cur_frame
            cur_frame_color = video.read()[1]
            cur_frame = cv2.cvtColor(cur_frame_color, cv2.COLOR_BGR2GRAY)

class OpticalFlowFeatures:
    def __init__(self, flow):
        self.flow = flow
    def get_hof(self, x_cells, y_cells, bins, density=True):
        hists = list()
        for x, y, _, _ in self.__iterate_cells(x_cells, y_cells):
            h, e = self.__hof(x, y, bins, density)
            hists.append(h)
        hists = np.array(hists).reshape(x_cells, y_cells, bins)
        return hists
    def get_magnitude(self, x_cells, y_cells):
        magnitude = np.zeros((x_cells, y_cells))
        for x, y, xi, yi in self.__iterate_cells(x_cells, y_cells):
            magnitude[xi, yi] = np.nanmean(np.sqrt(np.square(x) + np.square(y)))
        return magnitude
    def __iterate_cells(self, x_cells, y_cells):
        vectors = self.flow.vectors
        y_len, x_len, _ = vectors.shape
        x_cl = x_len / x_cells
        y_cl = y_len / y_cells
        for i in range(x_cells):
            for j in range(y_cells):
                x_base = i * x_cl
                y_base = j * y_cl
                x_components = vectors[y_base : y_base + y_cl, x_base : x_base + x_cl, 0]
                y_components = vectors[y_base : y_base + y_cl, x_base : x_base + x_cl, 1]
                yield x_components, y_components, i, j
    def __hof(self, x, y, bins, density=True):
        orientations = np.arctan2(x, y)
        magnitudes = np.sqrt(np.square(x) + np.square(y))
        hist, bin_edges = np.histogram(orientations, bins=bins,
            range=(-np.pi, np.pi), weights=magnitudes, density=density)
        return hist, bin_edges

if __name__ == "__main__":
    filepath = '/home/ljc/lijiancheng/cv_homework/hw2/dataset/KTH/person01_boxing_d1_uncomp.avi'
    for flow in OpticalFlow(filepath).farneback():
        flow.show()
