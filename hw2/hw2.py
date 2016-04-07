# coding=utf8

# export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
import os
import time
import pickle
import subprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

from video_features import VideoFeatures

class Solver(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset_path = '{}{sep}dataset{sep}{}{sep}'.format(os.getcwd(), dataset_name, sep=os.sep)
    def train(self):
        # load data
        train_videos, train_labels = self.__load_data('train_list.txt')
        train_features = self.__get_features(train_videos)
        for video in train_videos:
            train_features.append(VideoFeatures(video[0]).get_feature_vector())
        # train classifier
        classifier = OneVsRestClassifier(SVC())
        classifier.fit(train_features, train_labels)
        return classifier
    def test(self, classifier):
        # load data
        test_videos, test_labels = self.__load_data('test_list.txt')
        test_features = self.__get_features(train_videos)
        # predict
        predict_labels = classifier.predict(X)
        return accuracy_score(test_labels, predict_labels)
    def __load_data(self, filename):
        videos = list()
        labels = list()
        fd = open(os.path.join(self.dataset_path, filename), 'r')
        for line in fd:
            if self.dataset_name == 'KTH':
                line = line.split('\t')
                video_name, subsequences = line[0], line[1]
                video_name = video_name + '_uncomp.avi'
                subsequences = subsequences.split(',')
                for seq in subsequences:
                    seq = seq.strip().split('-')
                    videos.append([video_name, int(seq[0]), int(seq[1])])
                    labels.append(video_name.split('_')[1])
            else:
                # TODO
                videos.append(line.strip())
                labels.append(line.strip())
        fd.close()
        return videos, labels
    def __get_features(self, videos):
        features = list()
        for video in videos:
            video_path = self.dataset_path + video[0]
            features.append(VideoFeatures(video_path).get_feature_vector())
        return features
    # def __get_features(self, videos):
    #     for video in videos:
    #         video_path = self.dataset_path + video[0]
    #         command = r'./dense_trajectory_release_v1.2/release/DenseTrack ' + video_path
    #         if len(video) > 1:
    #             command += ' -S {} -E {}'.format(video[1], video[2])
    #         p = subprocess.Popen([command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    #         out, err = p.communicate()
    #         if err:
    #             print(err.decode())
    #         p.wait()
    #         out = out.decode()


if __name__ == "__main__":
    print(time.ctime())
    dataset_list = ['KTH', 'Youtube', 'Hollywood2']
    for dataset_name in dataset_list:
        print(dataset_name)
        solver = Solver(dataset_name)
        classifier = solver.train()
        # pickle.dump(classifier, open('classifier.pickle', 'wb'))
        # classifier = pickle.load(open('classifier.pickle', 'rb'))
        print(solver.test(classifier))
    
    print(time.ctime())
