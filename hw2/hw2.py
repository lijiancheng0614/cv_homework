# coding=utf8

# export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
import os
import time
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

from video_features import VideoFeatures

class Solver(object):
    def __init__(self, dataset_name, verbose=True):
        self.dataset_name = dataset_name
        self.dataset_path = '{}{sep}dataset{sep}{}{sep}'.format(os.getcwd(), dataset_name, sep=os.sep)
        self.verbose = verbose
    def train(self):
        if self.verbose:
            print('{} train...'.format(time.ctime()))
        # load data
        train_videos, train_labels = self.__load_data('train_list.txt')
        filepath = os.path.join(self.dataset_path, 'train_features.pickle')
        train_features = self.__get_features(train_videos, filepath)
        # train classifier
        classifier = OneVsRestClassifier(SVC())
        classifier.fit(train_features, train_labels)
        if self.verbose:
            print('{} train done.'.format(time.ctime()))
        return classifier
    def test(self, classifier):
        if self.verbose:
            print('{} test...'.format(time.ctime()))
        # load data
        test_videos, test_labels = self.__load_data('test_list.txt')
        filepath = os.path.join(self.dataset_path, 'test_features.pickle')
        test_features = self.__get_features(test_videos, filepath)
        # predict
        predict_labels = classifier.predict(test_features)
        if self.verbose:
            pickle.dump(predict_labels, open('predict_labels.pickle', 'wb'))
            print('{} test done.'.format(time.ctime()))
        return accuracy_score(test_labels, predict_labels)
    def __load_data(self, filename):
        if self.verbose:
            print('{} __load_data...'.format(time.ctime()))
        videos = list()
        labels = list()
        fd = open(os.path.join(self.dataset_path, filename), 'r')
        for line in fd:
            if self.dataset_name == 'KTH':
                line = line.split('\t')
                video_name = line[0]
                video_name = video_name + '_uncomp.avi'
                videos.append(video_name)
                labels.append(video_name.split('_')[1])
            else:
                # TODO
                line = line.strip()
                videos.append(line)
                labels.append(line)
        fd.close()
        if self.verbose:
            print('{} __load_data done. {} videos.'.format(time.ctime(), len(videos)))
        return videos, labels
    def __get_features(self, videos, filepath):
        if self.verbose:
            print('{} __get_features...'.format(time.ctime()))
        if os.path.exists(filepath):
            features = pickle.load(open(filepath, 'rb'))
        else:
            features = list()
            for video in videos:
                video_path = self.dataset_path + video
                features.append(VideoFeatures(video_path).get_feature_vector())
            pickle.dump(features, open(filepath, 'wb'))
        if self.verbose:
            print('{} __get_features done. features size: {} x {}.'.format(time.ctime(), len(features), len(features[0])))
        return features

if __name__ == "__main__":
    print(time.ctime())
    # dataset_list = ['KTH', 'Youtube', 'Hollywood2']
    dataset_list = ['KTH']
    for dataset_name in dataset_list:
        print('{} {}'.format(time.ctime(), dataset_name))
        solver = Solver(dataset_name)
        classifier = solver.train()
        pickle.dump(classifier, open('classifier.pickle', 'wb'))
        # classifier = pickle.load(open('classifier.pickle', 'rb'))
        print(solver.test(classifier))
    
    print(time.ctime())
