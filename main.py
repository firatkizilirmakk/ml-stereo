import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

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
    neg_offset_multiplier = 30 if pos_pair_dissim < 25 else 10

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

def create_dataset(dataset_path, block_size, is_test = False):
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

    # concatenate positive and negative feature vectors and corresponding labels
    feature_vectors = pos_feature_vectors + neg_feature_vectors
    labels = np.array([1]* len(pos_feature_vectors) + [-1] * len(neg_feature_vectors))

    # shuffle the lists
    z = list(zip(feature_vectors, labels))
    shuffle(z)
    feature_vectors, labels = zip(*z)

    feature_vectors = np.float32(feature_vectors)
    labels = np.array(labels)

    feature_path = ("test_" if is_test else "") + "features.npy"
    label_path = ("test_" if is_test else "") + "labels.npy"

    # save the feature vectors and labels for later use
    np.save(feature_path, feature_vectors)
    np.save(label_path, labels)

def read_data(data_path):
    data = np.load(data_path, allow_pickle = True)
    return data


def stereo(model, left_img, right_img, num_of_disparities, block_size):
    if left_img.shape != right_img.shape:
        return None

    height, width = left_img.shape
    # create new disparity image
    new_disp_img = np.zeros((height, width), dtype = np.int16)

    # loop over the image
    for y in range(block_size // 2, height - (block_size // 2)):
        for x in range(block_size // 2, width - (block_size // 2)):
            left_patch = np.int16(left_img[y - (block_size // 2): y + (block_size // 2) + 1, 
                                        x - (block_size // 2): x + (block_size // 2) + 1])

            min_score = 99999999
            min_score_disparity = -1

            # check the range from this x to x + num_of_disparities
            # put the most similar one
            for disparity in range(num_of_disparities):
                right_patch_x = x + disparity
                if right_patch_x < (width - (block_size // 2)):
                    right_patch = np.int16(right_img[y - (block_size // 2): y + (block_size // 2) + 1, 
                                            (right_patch_x) - (block_size // 2): (right_patch_x) + (block_size // 2) + 1])

                    # get the feature vector using left and right patches
                    feature_vector = np.float32(create_features(left_patch, right_patch, block_size))
                    feature_vector = feature_vector[np.newaxis, : ]

                    # predict whether these patches belong to the same place, object etc.
                    prediction = model.predict(feature_vector)
                    prediction = prediction[1][0][0]

                    # if patches belong the same place, calculate the score
                    if prediction == 1:
                        score = np.sum(feature_vector)

                        if score < min_score:
                            min_score = score
                            min_score_disparity = disparity   

            if min_score_disparity != -1:
                new_disp_img[y][x] = min_score_disparity * 8

    return new_disp_img

def stereoBayes(model, left_img, right_img, num_of_disparities, block_size):
    if left_img.shape != right_img.shape:
        return None

    height, width = left_img.shape
    # create new disparity image
    new_disp_img = np.zeros((height, width), dtype = np.int16)

    # loop over the image
    for y in range(block_size // 2, height - (block_size // 2)):
        for x in range(block_size // 2, width - (block_size // 2)):
            left_patch = np.int16(left_img[y - (block_size // 2): y + (block_size // 2) + 1,
                                        x - (block_size // 2): x + (block_size // 2) + 1])

            min_score = 99999999
            min_score_disparity = -1

            # check the range from this x to x + num_of_disparities
            # put the most similar one
            for disparity in range(num_of_disparities):
                right_patch_x = x + disparity
                if right_patch_x < (width - (block_size // 2)):
                    right_patch = np.int16(right_img[y - (block_size // 2): y + (block_size // 2) + 1,
                                            (right_patch_x) - (block_size // 2): (right_patch_x) + (block_size // 2) + 1])

                    # get the feature vector using left and right patches
                    feature_vector = np.float32(create_features(left_patch, right_patch, block_size))
                    feature_vector = feature_vector[np.newaxis, : ]

                    # predict whether these patches belong to the same place, object etc.
                    prediction = model.predictProb(feature_vector)
                    prediction = prediction[1][0][0]

                    # if patches belong the same place, calculate the score
                    if prediction == 1:
                        score = np.sum(feature_vector)

                        if score < min_score:
                            min_score = score
                            min_score_disparity = disparity

            if min_score_disparity != -1:
                new_disp_img[y][x] = min_score_disparity * 8

    return new_disp_img

from random import shuffle

block_size = 7
testing = True
model_name = "SVM"

if testing:
    # testing part
    num_of_disparities = 19
    if model_name == "Bayes":
        model = cv2.ml.NormalBayesClassifier_load("bayes_model")
    elif model_name == "SVM":
        model = cv2.ml.SVM_load("svm_model")

    test_dataset_path = "test_dataset"

    # create the test feature vectors and labels
    # create_dataset(test_dataset_path, block_size, is_test=True)

    # predict for whole test set to get confusion matrix and other scores
    X = read_data("test_features.npy")
    y = np.float32(read_data("test_labels.npy"))

    preds = model.predict(X)[1]

    cm = confusion_matrix(y, preds)
    (prec, recall, f1, _) = precision_recall_fscore_support(y, preds, average="binary")

    print(cm)
    print("Precision : ", prec, " Recall : ", recall, " F1 : ", f1)

    dataset_dir = os.listdir(test_dataset_path)
    for directory in dataset_dir:
        img_path = os.path.join(test_dataset_path, directory)

        left_img = cv2.imread(img_path + "/im6.ppm", 0)
        right_img = cv2.imread(img_path + "/im2.ppm", 0)
        disp_img = cv2.imread(img_path + "/disp6.pgm", 0)

        if model_name == "Bayes":
            predicted_disp_img = stereoBayes(model, left_img, right_img, num_of_disparities, block_size)
        elif model_name == "SVM":
            predicted_disp_img = stereo(model, left_img, right_img, num_of_disparities, block_size)

        mse = (np.square(disp_img - predicted_disp_img)).mean(axis = None)
        print("MSE : ", mse)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(disp_img, "gray")
        ax[1].imshow(predicted_disp_img, "gray")

        ax[0].set_xlabel("Ground Truth Disparity Map")
        ax[1].set_xlabel("Calculated Disparity Map")
        plt.show()

else:
    # training part
    dataset_path = "train_dataset"

    create_dataset(dataset_path, block_size)

    X = read_data("features.npy")
    y = np.float32(read_data("labels.npy"))

    if model_name == "Bayes":
        bayes = cv2.ml.NormalBayesClassifier_create()
        bayes.train(X, cv2.ml.ROW_SAMPLE, y)
        bayes.save("bayes_model")
    elif model_name == "SVM":
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

        svm.train(X, cv2.ml.ROW_SAMPLE, y)
        svm.save("svm_model")