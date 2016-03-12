import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def demo1():
    # setting
    input_filepath = os.path.join('in', '1.jpg')
    if not os.path.isdir('out'):
        os.makedirs('out')
    output_filepath = ['demo1_B.txt', 'demo1_G.txt', 'demo1_R.txt', \
        'demo1_H.txt', 'demo1_S.txt', 'demo1_V.txt', \
        'demo1_L.txt', 'demo1_a-.txt', 'demo1_b-.txt']
    output_filepath = [os.path.join('out', i) for i in output_filepath]
    # process BGR
    img = cv2.imread(input_filepath)
    for k in range(3):
        fd = open(output_filepath[k], 'w')
        for i in img[:, :, k]:
            fd.write(' '.join(['{:3}'.format(j) for j in i]) + '\n')
        fd.close()
    # process HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for k in range(3):
        fd = open(output_filepath[k + 3], 'w')
        for i in imgHSV[:, :, k]:
            fd.write(' '.join(['{:3}'.format(j) for j in i]) + '\n')
        fd.close()
    # process CIELab
    imgLab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    for k in range(3):
        fd = open(output_filepath[k + 6], 'w')
        for i in imgLab[:, :, k]:
            fd.write(' '.join(['{:3}'.format(j) for j in i]) + '\n')
        fd.close()

def demo2():
    # setting
    input_filepath = ['r1.jpg', 'r2.jpg', 's1.jpg', 's2.jpg']
    input_filepath = [os.path.join('in', i) for i in input_filepath]
    # process
    for i in input_filepath:
        # process BGR
        img = cv2.imread(i)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = [cv2.calcHist([imgRGB], [k], None, [256], [0, 256]) for k in range(3)]
        x = np.arange(256) + 0.5
        plt.subplot(221), plt.imshow(imgRGB)
        plt.subplot(222), plt.bar(x, hist[0], color = 'r', edgecolor = 'r')
        plt.subplot(223), plt.bar(x, hist[1], color = 'g', edgecolor = 'g')
        plt.subplot(224), plt.bar(x, hist[2], color = 'b', edgecolor = 'b')
        plt.show()
        # process HSV
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = [cv2.calcHist([imgHSV], [0], None, [180], [0, 180])]
        hist.append(cv2.calcHist([imgHSV], [1], None, [256], [0, 256]))
        plt.subplot(211), plt.bar(np.arange(180) + 0.5, hist[0])
        plt.subplot(212), plt.bar(x, hist[1])
        plt.show()
        # process CIELab
        imgLab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        hist = []
        for k in range(1, 3):
            hist.append(cv2.calcHist([imgLab], [k], None, [50], [0, 256]))
        x = np.arange(50) + 0.5
        plt.subplot(211), plt.bar(x, hist[0])
        plt.subplot(212), plt.bar(x, hist[1])
        plt.show()

def demo3_process(fd, hist, methods, caption, LATEX_STYLE = False):
    for i in range(2):
        for k in range(len(hist[i])):
            cv2.normalize(hist[i][k], hist[i][k]).flatten()
    METHOD_COUNT = len(methods)
    if LATEX_STYLE:
        fd.write(r'\begin{table}[h!]')
        fd.write('\n')
        fd.write(r'    \centering')
        fd.write('\n')
        fd.write('    \caption{{{}}}'.format(caption))
        fd.write('\n')
        fd.write(r'    \begin{tabular}{ccccc}')
        fd.write('\n')
        fd.write(r'        methods & Correlation & Intersection & Chi-Square & Bhattacharyya \\ \hline')
        fd.write('\n')
        for k in range(len(hist[0])):
            d = [cv2.compareHist(hist[0][k], hist[1][k], methods[i]) for i in range(METHOD_COUNT)]
            fd.write('        {} & {:.4} & {:.4} & {:.4} & {:.4} \\\\'.format(k, d[0], d[1], d[2], d[3]))
            fd.write('\n')
        fd.write(r'    \end{tabular}')
        fd.write('\n')
        fd.write(r'\end{table}')
        fd.write('\n')
    else:
        fd.write('{}\n'.format(caption))
        for k in range(len(hist[0])):
            fd.write('{}'.format(k))
            for i in range(METHOD_COUNT):
                d = cv2.compareHist(hist[0][k], hist[1][k], methods[i])
                fd.write(' {:.4}'.format(d))
            fd.write('\n')
        fd.write('\n')

def demo3():
    # setting
    input_filepath = ['r1.jpg', 'r2.jpg', 's1.jpg', 's2.jpg']
    input_filepath = [os.path.join('in', i) for i in input_filepath]
    compare_pair_list = [(0, 0), (0, 1), (2, 3), (0, 2), (1, 3)]
    methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_CHISQR, cv2.HISTCMP_BHATTACHARYYA]
    if not os.path.isdir('out'):
        os.makedirs('out')
    output_filepath = os.path.join('out', 'demo3.txt')
    LATEX_STYLE = True
    # process
    fd = open(output_filepath, 'w')
    for a, b in compare_pair_list:
        # process BGR
        img = [cv2.imread(input_filepath[a]), cv2.imread(input_filepath[b])]
        hist = [[cv2.calcHist([img[i]], [k], None, [256], [0, 256]) \
            for k in range(3)] for i in range(2)]
        demo3_process(fd, hist, methods, \
            '{}, {}'.format(input_filepath[a].split('\\')[-1], input_filepath[b].split('\\')[-1]), LATEX_STYLE)
    fd.write('\n\n')
    for a, b in compare_pair_list:
        # process HSV
        img = [cv2.imread(input_filepath[a]), cv2.imread(input_filepath[b])]
        imgHSV = [cv2.cvtColor(i, cv2.COLOR_BGR2HSV) for i in img]
        hist = [[cv2.calcHist([imgHSV[i]], [0], None, [180], [0, 180])] \
            for i in range(2)]
        for i in range(2):
            hist[i].append(cv2.calcHist([imgHSV[i]], [1], None, [256], [0, 256]))
        demo3_process(fd, hist, methods, \
            '{}, {}'.format(input_filepath[a].split('\\')[-1], input_filepath[b].split('\\')[-1]), LATEX_STYLE)
    fd.write('\n\n')
    for a, b in compare_pair_list:
        # process CIELab
        img = [cv2.imread(input_filepath[a]), cv2.imread(input_filepath[b])]
        imgLab = [cv2.cvtColor(i, cv2.COLOR_BGR2Lab) for i in img]
        hist = [[] for i in range(2)]
        for i in range(2):
            for k in range(1, 3):
                hist[i].append(cv2.calcHist([imgLab[i]], [k], None, [50], [0, 256]))
        demo3_process(fd, hist, methods, \
            '{}, {}'.format(input_filepath[a].split('\\')[-1], input_filepath[b].split('\\')[-1]), LATEX_STYLE)
    fd.write('\n\n')
    for a, b in compare_pair_list:
        # process RGB cumulative histogram
        img = [cv2.imread(input_filepath[a]), cv2.imread(input_filepath[b])]
        hist = [[cv2.calcHist([img[i]], [k], None, [256], [0, 256]).cumsum() \
            for k in range(3)] for i in range(2)]
        demo3_process(fd, hist, methods, \
            '{}, {}'.format(input_filepath[a].split('\\')[-1], input_filepath[b].split('\\')[-1]), LATEX_STYLE)
    fd.write('\n\n')
    fd.close()

# demo1()
# demo2()
demo3()