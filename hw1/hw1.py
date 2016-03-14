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
    imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for k in range(3):
        fd = open(output_filepath[k + 6], 'w')
        for i in imgLAB[:, :, k]:
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
        imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hist = []
        for k in range(1, 3):
            hist.append(cv2.calcHist([imgLAB], [k], None, [50], [0, 256]))
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

def demo3_BGR(compare_pair_list, input_filepath, fd, methods, LATEX_STYLE):
    # process BGR histogram
    for a, b in compare_pair_list:
        img = [cv2.imread(input_filepath[a]), cv2.imread(input_filepath[b])]
        hist = [[cv2.calcHist([img[i]], [k], None, [256], [0, 256]) \
            for k in range(3)] for i in range(2)]
        demo3_process(fd, hist, methods, \
            '{}, {}'.format(input_filepath[a].split(os.sep)[-1], input_filepath[b].split(os.sep)[-1]), LATEX_STYLE)
    fd.write('\n\n')

def demo3_HSV(compare_pair_list, input_filepath, fd, methods, LATEX_STYLE):
    # process HSV histogram
    for a, b in compare_pair_list:
        img = [cv2.imread(input_filepath[a]), cv2.imread(input_filepath[b])]
        imgHSV = [cv2.cvtColor(i, cv2.COLOR_BGR2HSV) for i in img]
        hist = [[cv2.calcHist([imgHSV[i]], [0], None, [180], [0, 180])] \
            for i in range(2)]
        for i in range(2):
            hist[i].append(cv2.calcHist([imgHSV[i]], [1], None, [256], [0, 256]))
        demo3_process(fd, hist, methods, \
            '{}, {}'.format(input_filepath[a].split(os.sep)[-1], input_filepath[b].split(os.sep)[-1]), LATEX_STYLE)
    fd.write('\n\n')

def demo3_LAB(compare_pair_list, input_filepath, fd, methods, LATEX_STYLE):
    # process CIELab histogram
    for a, b in compare_pair_list:
        img = [cv2.imread(input_filepath[a]), cv2.imread(input_filepath[b])]
        imgLAB = [cv2.cvtColor(i, cv2.COLOR_BGR2LAB) for i in img]
        hist = [[] for i in range(2)]
        for i in range(2):
            for k in range(1, 3):
                hist[i].append(cv2.calcHist([imgLAB[i]], [k], None, [50], [0, 256]))
        demo3_process(fd, hist, methods, \
            '{}, {}'.format(input_filepath[a].split(os.sep)[-1], input_filepath[b].split(os.sep)[-1]), LATEX_STYLE)
    fd.write('\n\n')

def demo3_BGRch(compare_pair_list, input_filepath, fd, methods, LATEX_STYLE):
    # process RGB cumulative histogram
    for a, b in compare_pair_list:
        img = [cv2.imread(input_filepath[a]), cv2.imread(input_filepath[b])]
        hist = [[cv2.calcHist([img[i]], [k], None, [256], [0, 256]).cumsum() \
            for k in range(3)] for i in range(2)]
        demo3_process(fd, hist, methods, \
            '{}, {}'.format(input_filepath[a].split(os.sep)[-1], input_filepath[b].split(os.sep)[-1]), LATEX_STYLE)
    fd.write('\n\n')

def L1_distance(a, b):
    # return 2d-array L1 distance
    l1_distance = 0
    for i in range(len(a)):
        for j in range(len(a[i])):
            l1_distance += abs(a[i][j] - b[i][j])
    return l1_distance

def demo3_CCV(input_filepath, fd, LATEX_STYLE):
    # process comparing CCV with L1 distance
    from ccv import CCV
    caption = 'CCV-L1'
    if LATEX_STYLE:
        fd.write(r'\begin{table}[h!]')
        fd.write('\n')
        fd.write(r'    \centering')
        fd.write('\n')
        fd.write('    \caption{{{}}}'.format(caption))
        fd.write('\n')
        fd.write(r'    \begin{tabular}{ccccc}')
        fd.write('\n')
        fd.write(r'         & r1 & r2 & s1 & s2 \\ \hline')
        fd.write('\n')
    else:
        fd.write('{}\n'.format(caption))
        fd.write('{:8} {:8} {:8} {:8} {:8}\n'.format('', 'r1', 'r2', 's1', 's2'))
    for a in range(len(input_filepath)):
        if LATEX_STYLE:
            fd.write('        {}'.format(input_filepath[a].split(os.sep)[-1].rstrip('.jpg')))
        else:
            fd.write('{:8}'.format(input_filepath[a].split(os.sep)[-1].rstrip('.jpg')))
        for b in range(len(input_filepath)):
            ccv = [CCV(input_filepath[a]).get_vector(), \
                CCV(input_filepath[b]).get_vector()]
            l1_distance = L1_distance(ccv[0], ccv[1])
            if LATEX_STYLE:
                fd.write(' & {}'.format(l1_distance))
            else:
                fd.write('{:8}'.format(l1_distance))
        if LATEX_STYLE:
            fd.write(r' \\ \hline')
        fd.write('\n')
    if LATEX_STYLE:
        fd.write(r'    \end{tabular}')
        fd.write('\n')
        fd.write(r'\end{table}')
    fd.write('\n\n')

def demo3_CRH(input_filepath, fd, LATEX_STYLE):
    # process comparing Centering Refinement RGB histogram with L1 distance
    from crh import CRH
    caption = 'Centering Refinement Histogram-L1'
    if LATEX_STYLE:
        fd.write(r'\begin{table}[h!]')
        fd.write('\n')
        fd.write(r'    \centering')
        fd.write('\n')
        fd.write('    \caption{{{}}}'.format(caption))
        fd.write('\n')
        fd.write(r'    \begin{tabular}{ccccc}')
        fd.write('\n')
        fd.write(r'         & r1 & r2 & s1 & s2 \\ \hline')
        fd.write('\n')
    else:
        fd.write('{}\n'.format(caption))
        fd.write('{:8} {:8} {:8} {:8} {:8}\n'.format('', 'r1', 'r2', 's1', 's2'))
    for a in range(len(input_filepath)):
        if LATEX_STYLE:
            fd.write('        {}'.format(input_filepath[a].split(os.sep)[-1].rstrip('.jpg')))
        else:
            fd.write('{:8}'.format(input_filepath[a].split(os.sep)[-1].rstrip('.jpg')))
        for b in range(len(input_filepath)):
            crh = [CRH(input_filepath[a]).get_vector(), \
                CRH(input_filepath[b]).get_vector()]
            l1_distance = L1_distance(crh[0], crh[1])
            if LATEX_STYLE:
                fd.write(' & {}'.format(l1_distance))
            else:
                fd.write('{:8}'.format(l1_distance))
        if LATEX_STYLE:
            fd.write(r' \\ \hline')
        fd.write('\n')
    if LATEX_STYLE:
        fd.write(r'    \end{tabular}')
        fd.write('\n')
        fd.write(r'\end{table}')
    fd.write('\n\n')

def demo3_CCDR(input_filepath, fd, LATEX_STYLE):
    # process comparing Color Coherence Distance Refinement with L1 distance
    from ccdr import CCDR
    caption = 'Color Coherence Distance Refinement-L1'
    if LATEX_STYLE:
        fd.write(r'\begin{table}[h!]')
        fd.write('\n')
        fd.write(r'    \centering')
        fd.write('\n')
        fd.write('    \caption{{{}}}'.format(caption))
        fd.write('\n')
        fd.write(r'    \begin{tabular}{ccccc}')
        fd.write('\n')
        fd.write(r'         & r1 & r2 & s1 & s2 \\ \hline')
        fd.write('\n')
    else:
        fd.write('{}\n'.format(caption))
        fd.write('{:8} {:8} {:8} {:8} {:8}\n'.format('', 'r1', 'r2', 's1', 's2'))
    for a in range(len(input_filepath)):
        if LATEX_STYLE:
            fd.write('        {}'.format(input_filepath[a].split(os.sep)[-1].rstrip('.jpg')))
        else:
            fd.write('{:8}'.format(input_filepath[a].split(os.sep)[-1].rstrip('.jpg')))
        for b in range(len(input_filepath)):
            ccdr = [CCDR(input_filepath[a]).get_vector(), \
                CCDR(input_filepath[b]).get_vector()]
            l1_distance = L1_distance(ccdr[0], ccdr[1])
            if LATEX_STYLE:
                fd.write(' & {}'.format(l1_distance))
            else:
                fd.write('{:8}'.format(l1_distance))
        if LATEX_STYLE:
            fd.write(r' \\ \hline')
        fd.write('\n')
    if LATEX_STYLE:
        fd.write(r'    \end{tabular}')
        fd.write('\n')
        fd.write(r'\end{table}')
    fd.write('\n\n')

def demo3():
    # setting
    input_filepath = ['r1.jpg', 'r2.jpg', 's1.jpg', 's2.jpg']
    input_filepath = [os.path.join('in', i) for i in input_filepath]
    compare_pair_list = [(0, 0), (0, 1), (2, 3), (0, 2), (1, 3)]
    methods = [cv2.cv.CV_COMP_CORREL, cv2.cv.CV_COMP_INTERSECT, \
        cv2.cv.CV_COMP_CHISQR, cv2.cv.CV_COMP_BHATTACHARYYA]
    if not os.path.isdir('out'):
        os.makedirs('out')
    output_filepath = os.path.join('out', 'demo3.txt')
    LATEX_STYLE = True
    # process
    fd = open(output_filepath, 'w')
    demo3_BGR(compare_pair_list, input_filepath, fd, methods, LATEX_STYLE)
    demo3_HSV(compare_pair_list, input_filepath, fd, methods, LATEX_STYLE)
    demo3_LAB(compare_pair_list, input_filepath, fd, methods, LATEX_STYLE)
    demo3_BGRch(compare_pair_list, input_filepath, fd, methods, LATEX_STYLE)
    demo3_CCV(input_filepath, fd, LATEX_STYLE)
    demo3_CRH(input_filepath, fd, LATEX_STYLE)
    demo3_CCDR(input_filepath, fd, LATEX_STYLE)
    fd.close()

def compare_hist(query_image, image_set, hist_type, method):
    img = cv2.imread(query_image)
    img_set = [cv2.imread(i) for i in image_set]
    if hist_type == 0:
        # BGR
        hist = [cv2.calcHist([img], [k], None, [256], [0, 256]) for k in range(3)]
        for i in hist:
            cv2.normalize(i, i)
        hist = np.mean(hist, 0)
        hist_set = [[cv2.calcHist([i], [k], None, [256], [0, 256]) for k in range(3)] \
            for i in img_set]
        for i in hist_set:
            for j in i:
                cv2.normalize(j, j)
        hist_set = [np.mean(i, 0) for i in hist_set]
    elif hist_type == 1:
        # HSV
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgHSV_set = [cv2.cvtColor(i, cv2.COLOR_BGR2HSV) for i in img_set]
        hist = cv2.calcHist([imgHSV], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hist_set = [cv2.calcHist([i], [0, 1], None, [8, 8], [0, 180, 0, 256]) \
            for i in imgHSV_set]
        for i in hist_set:
            cv2.normalize(i, i)
    else:
        # BGR cumulative
        hist = [cv2.calcHist([img], [k], None, [256], [0, 256]).cumsum() for k in range(3)]
        for i in hist:
            cv2.normalize(i, i)
        hist = np.mean(hist, 0)
        hist_set = [[cv2.calcHist([i], [k], None, [256], [0, 256]).cumsum() for k in range(3)] \
            for i in img_set]
        for i in hist_set:
            for j in i:
                cv2.normalize(j, j)
        hist_set = [np.mean(i, 0) for i in hist_set]
    d = [cv2.compareHist(hist, i, method) for i in hist_set]
    if method == cv2.cv.CV_COMP_CORREL or method == cv2.cv.CV_COMP_INTERSECT:
        return np.argmax(d)
    return np.argmin(d)

def CBIR(query_image, image_set, method):
    comp_method = [cv2.cv.CV_COMP_CORREL, cv2.cv.CV_COMP_INTERSECT, \
        cv2.cv.CV_COMP_CHISQR, cv2.cv.CV_COMP_BHATTACHARYYA]
    if method < 12:
        return compare_hist(query_image, image_set, method // 4, comp_method[method % 4])
    elif method == 12:
        from ccv import CCV
        ccv = CCV(query_image).get_vector()
        ccv_set = [CCV(i).get_vector() for i in image_set]
        l1_distance = [L1_distance(ccv, i) for i in ccv_set]
        return np.argmin(l1_distance)
    elif method == 13:
        from crh import CRH
        crh = CRH(query_image).get_vector()
        crh_set = [CRH(i).get_vector() for i in image_set]
        l1_distance = [L1_distance(crh, i) for i in crh_set]
        return np.argmin(l1_distance)
    elif method == 14:
        from ccdr import CCDR
        ccdr = CCDR(query_image).get_vector()
        ccdr_set = [CCDR(i).get_vector() for i in image_set]
        l1_distance = [L1_distance(ccdr, i) for i in ccdr_set]
        return np.argmin(l1_distance)
    return 0

def demo4():
    # setting
    input_filepath = ['r1.jpg', 'r2.jpg', 's1.jpg', 's2.jpg']
    input_filepath = [os.path.join('in', i) for i in input_filepath]
    methods = [cv2.cv.CV_COMP_CORREL, cv2.cv.CV_COMP_INTERSECT, \
        cv2.cv.CV_COMP_CHISQR, cv2.cv.CV_COMP_BHATTACHARYYA]
    if not os.path.isdir('out'):
        os.makedirs('out')
    output_filepath = os.path.join('out', 'demo4.txt')
    LATEX_STYLE = True
    # process
    fd = open(output_filepath, 'w')
    caption = 'CBIR'
    if LATEX_STYLE:
        fd.write(r'\begin{table}[h!]')
        fd.write('\n')
        fd.write(r'    \centering')
        fd.write('\n')
        fd.write('    \caption{{{}}}'.format(caption))
        fd.write('\n')
        fd.write(r'    \begin{tabular}{ccccc}')
        fd.write('\n')
        fd.write(r'         & r1 & r2 & s1 & s2 \\ \hline')
        fd.write('\n')
    else:
        fd.write('{}\n'.format(caption))
        fd.write('   r1 r2 r3 r4\n')
    for method in range(15):
        if LATEX_STYLE:
            fd.write('        {:2}'.format(method))
        else:
            fd.write('{:2}'.format(method))
        for k in range(len(input_filepath)):
            query_image = input_filepath[k]
            image_set = [i for i in input_filepath]
            image_set.pop(k)
            i = CBIR(query_image, image_set, method)
            i = image_set[i].split(os.sep)[-1].rstrip('.jpg')
            if LATEX_STYLE:
                fd.write(' & {}'.format(i))
            else:
                fd.write(' {}'.format(i))
        if LATEX_STYLE:
            fd.write(r' \\ \hline')
        fd.write('\n')
    if LATEX_STYLE:
        fd.write(r'    \end{tabular}')
        fd.write('\n')
        fd.write(r'\end{table}')
    fd.write('\n\n')
    fd.close()

demo1()
demo2()
demo3()
demo4()