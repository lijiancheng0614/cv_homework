# coding=utf-8

import argparse
import cv2
from glob import glob
import numpy as np
import os

def set_arguments():
    parser = argparse.ArgumentParser(description='Camera calibration demo')
    parser.add_argument('--img_mask', default='./data/left*.jpg')
    parser.add_argument('--output_dir', default='./output/')
    parser.add_argument('--square_size', type=float, default=1.0)
    return parser

def get_undistorted_image(img, camera_matrix, dist_coefs):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
    dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

if __name__ == '__main__':
    parser = set_arguments()
    cmd_args = parser.parse_args()
    img_names = glob(cmd_args.img_mask)
    if not os.path.isdir(cmd_args.output_dir):
        os.mkdir(cmd_args.output_dir)

    pattern_size = (9, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= cmd_args.square_size

    obj_points = list()
    img_points = list()
    img_names_undistort = list()
    # find the corners
    for filename in img_names:
        img = cv2.imread(filename, 0)
        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if not found:
            print('{}: chessboard not found'.format(filename))
            continue
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1))
        img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(img_vis, pattern_size, corners, found)
        name = os.path.splitext(os.path.split(filename)[1])[0]
        outfile = cmd_args.output_dir + name + '_chess.png'
        cv2.imwrite(outfile, img_vis)
        img_names_undistort.append(outfile)
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    # intrinsic parameters
    print('camera matrix:\n{}'.format(camera_matrix))
    # distortion cofficients = (k_1, k_2, p_1, p_2, k_3)
    print('distortion coefficients:\n{}'.format(dist_coefs.ravel()))
    for i, filename in enumerate(img_names):
        print('{}'.format(filename))
        # rotation vector
        print('rotation vector: {}.T'.format(rvecs[i].ravel()))
        # translation vector
        print('translation vector: {}.T'.format(tvecs[i].ravel()))

    # undistortion
    for filename in img_names_undistort:
        img = cv2.imread(filename)
        undistorted_img = get_undistorted_image(img, camera_matrix, dist_coefs)
        name = os.path.splitext(os.path.split(filename)[1])[0]
        outfile = cmd_args.output_dir + name + '_undistorted.png'
        cv2.imwrite(outfile, undistorted_img)
