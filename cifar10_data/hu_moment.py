import pickle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys

import random
random.seed(2021)
sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def calculate_humoments(train_images, train_labels, calib_file_path=None):

    classes = []
    HuMoments_1 = []
    HuMoments_2 = []
    calib_HuMoments_1 = []
    calib_HuMoments_2 = []
    for i in range(len(train_labels)):
        image = train_images[i]
        cls = train_labels[i]
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        moments = cv.moments(image)
        HuMoments = cv.HuMoments(moments)
        classes.append(cls)
        HuMoments_1.append(HuMoments[0])
        HuMoments_2.append(HuMoments[1])

    for file in os.listdir(calib_file_path):
        print(file)
        srcImg = cv.imread(calib_file_path + '/' + str(file))
        image = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)
        moments = cv.moments(image)
        HuMoments = cv.HuMoments(moments)
        calib_HuMoments_1.append(HuMoments[0])
        calib_HuMoments_2.append(HuMoments[1])

    return classes, HuMoments_1, HuMoments_2, calib_HuMoments_1, calib_HuMoments_2


def plot_humoments(train_images, train_labels, calib_file_path=None, plot_idx=None, output_file=None):

    train_cls_list = []
    train_plot_hu_moments_1 = []
    train_plot_hu_moments_2 = []
    for i in range(10):
        train_cls_list.append([])
        train_plot_hu_moments_1.append([])
        train_plot_hu_moments_2.append([])

    cls_list, hu_moments_1, hu_moments_2, calib_HuMoments_1, calib_HuMoments_2 = \
        calculate_humoments(train_images=train_images, train_labels=train_labels, calib_file_path=calib_file_path)

    for j, cls in enumerate(cls_list):
        train_cls_list[cls[0]].append(cls[0])
        train_plot_hu_moments_1[cls[0]].append(hu_moments_1[j])
        train_plot_hu_moments_2[cls[0]].append(hu_moments_2[j])
        if (j % 1000 == 0):
            print('Finished %ik of %ik images' % (j / 1000, 50000 / 1000))
    train_plot_list = [train_cls_list, train_plot_hu_moments_1, train_plot_hu_moments_2]
    calib_plot_list = [calib_HuMoments_1, calib_HuMoments_2]


    r = lambda: random.randint(0, 255)
    colors=[]
    for i in range(11):
        color = ('#%02X%02X%02X' % (r(),r(),r()))
        colors.append(color)
    area = np.pi * 2**2

    if plot_idx is None:
        for i in range(len(train_plot_list[0])):
            cls = train_plot_list[0][i][0]
            train_x = np.log(np.abs(train_plot_list[1][i]))  # hu_moments_1
            train_y = np.log(np.abs(train_plot_list[2][i]))  # hu_moments_1
            plt.scatter(train_x, train_y, s=area, c=colors[(cls)], alpha=0.1, label=cls)

    else:
        cls = train_plot_list[0][plot_idx][0]
        train_x = np.log(np.abs(train_plot_list[1][plot_idx])) # hu_moments_1
        train_y = np.log(np.abs(train_plot_list[2][plot_idx])) # hu_moments_2
        plt.scatter(train_x, train_y, s=area, c=colors[cls], alpha=0.1, label='cls_num:' + str(cls))

    calib_x = np.log(np.abs(calib_plot_list[0]))  # hu_moments_1
    calib_y = np.log(np.abs(calib_plot_list[1]))  # hu_moments_2
    plt.scatter(calib_x, calib_y, s=area, marker='x', c='red', alpha=0.5)
    plt.plot([np.min(calib_x), np.max(calib_x)], [np.min(calib_y), np.min(calib_y)], c='red')  # buttom
    plt.plot([np.min(calib_x), np.max(calib_x)], [np.max(calib_y), np.max(calib_y)], c='red')  # top
    plt.plot([np.min(calib_x), np.min(calib_x)], [np.min(calib_y), np.max(calib_y)], c='red')  # left
    plt.plot([np.max(calib_x), np.max(calib_x)], [np.min(calib_y), np.max(calib_y)], c='red')  # right

    plt.xlabel('hu_moments_1')
    plt.ylabel('hu_moments_2')
    plt.legend()
    plt.savefig(output_file, dpi=600)
    plt.show()

    return

if __name__ == '__main__':
    trainset_path = 'trainset/'
    data1 = unpickle(trainset_path + 'data_batch_1')
    data2 = unpickle(trainset_path + 'data_batch_2')
    data3 = unpickle(trainset_path + 'data_batch_3')
    data4 = unpickle(trainset_path + 'data_batch_4')
    data5 = unpickle(trainset_path + 'data_batch_5')

    images_1 = np.array(data1['data']).reshape(-1, 32, 32, 3)
    images_2 = np.array(data2['data']).reshape(-1, 32, 32, 3)
    images_3 = np.array(data3['data']).reshape(-1, 32, 32, 3)
    images_4 = np.array(data4['data']).reshape(-1, 32, 32, 3)
    images_5 = np.array(data5['data']).reshape(-1, 32, 32, 3)

    labels_1 = np.array(data1['labels'])
    labels_2 = np.array(data2['labels'])
    labels_3 = np.array(data3['labels'])
    labels_4 = np.array(data4['labels'])
    labels_5 = np.array(data5['labels'])

    images = np.vstack((images_1, images_2, images_3, images_4, images_5))
    labels = np.vstack((labels_1, labels_2, labels_3, labels_4, labels_5))
    calib_file_path = 'calib_dataset_40/'
    output_file = 'calib_dataset_40' + '.png'
    plot_humoments(images.reshape(50000,32,32,3), labels.reshape(50000,1), plot_idx=None, \
                   calib_file_path=calib_file_path, output_file=output_file)