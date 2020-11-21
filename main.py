import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def get_positive_pair(left_img, right_img, disp_img, x, y, block_size):
    if left_img.shape != right_img.shape:
        return None

    # starting from (block_size // 2), (block_size // 2) instead of 0,0
    if x < (block_size // 2) or y < (block_size // 2):
        return None

    height, width = left_img.shape
    gt_disparity = disp_img[y][x] // 8

    # mathced block can not be found
    if x + gt_disparity >= (width - (block_size // 2)):
        return None

    # get left and right patch wrt x, y and ground truth disparity
    left_patch = np.int16(left_img[y - (block_size // 2): y + (block_size // 2) + 1, x - (block_size // 2): x + (block_size // 2) + 1])
    right_patch = np.int16(right_img[y - (block_size // 2): y + (block_size // 2) + 1, (x + gt_disparity)  - (block_size // 2): (x + gt_disparity) + (block_size // 2) + 1])

    # dissimilarity is the norm of difference of the patches. more similar pathces, less dissimilarity
    dissimilarity = np.linalg.norm(left_patch - right_patch)

    return left_patch, right_patch, dissimilarity

def get_negative_pair(left_patch, right_img, y, block_size, pos_pair_dissim):
    # starting from (block_size // 2), (block_size // 2) instead of 0,0
    if y < (block_size // 2):
        return None

    height, width = right_img.shape
    x = block_size // 2
    dissimilarity = 0

    # determine negative multiplier, very big for similar patches
    neg_offset_multiplier = 30 if pos_dissim < 25 else 10

    while(dissimilarity < (pos_pair_dissim * neg_offset_multiplier)):
        right_patch = np.int16(right_img[y - (block_size // 2): y + (block_size // 2) + 1, x - (block_size // 2): x + (block_size // 2) + 1])

        # dissimilarity is the norm of difference of the patches. more similar pathces, less dissimilarity
        dissimilarity = np.linalg.norm(left_patch - right_patch)
        x += 1

        # to avoid infinite loop
        if x >= width - (block_size // 2):
            return None

    return left_patch, right_patch, dissimilarity

def create_features(left_patch, right_patch, block_size):
    features = np.int16(abs(left_patch - right_patch))
    feature_vector = np.reshape(features, (block_size * block_size))

    return feature_vector

block_size = 7
dataset_path = "dataset"
dataset_dir = os.listdir(dataset_path)

pos_feature_vectors = []
neg_feature_vectors = []

# loop over the image directories
for img_dir in dataset_dir:
    img_path = os.path.join(dataset_path, img_dir)

    left_img = cv2.imread(img_path + "/im6.ppm", 0)
    right_img = cv2.imread(img_path + "/im2.ppm", 0)
    disp_img = cv2.imread(img_path + "/disp6.pgm", 0)

    # loop over the image
    height, width = left_img.shape
    for y in range(block_size // 2, height - (block_size // 2)):
        for x in range(block_size // 2, width - (block_size // 2)):

            # get positive pair around x,y
            positive_pair = get_positive_pair(left_img, right_img, disp_img, x, y, block_size)
            if positive_pair is not None:
                pos_left_patch, pos_right_patch, pos_dissim = positive_pair
                # get negative pair around x,y
                negative_pair = get_negative_pair(pos_left_patch, right_img, y, block_size, pos_dissim)

                if negative_pair is not None:
                    neg_left_patch, neg_right_patch, neg_dissim = negative_pair

                    # create feature vectors from patches
                    pos_features = create_features(pos_left_patch, pos_right_patch, block_size)
                    neg_features = create_features(pos_left_patch, neg_right_patch, block_size)

                    pos_feature_vectors.append(pos_features)
                    neg_feature_vectors.append(neg_features)
        break

# concatenate positive and negative feature vectors and corresponding labels
feature_vectors = pos_feature_vectors + neg_feature_vectors
labels = [1]* len(pos_feature_vectors) + [-1] * len(neg_feature_vectors)

# split the data into train and test parts
trainX, testX, trainY, testY = train_test_split(feature_vectors, labels, shuffle = True, test_size = 0.2)