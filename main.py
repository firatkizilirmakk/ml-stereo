import sys
import os
import cv2
import numpy as np
import pickle 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz

# import models
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

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

def create_features2(left_patch, right_patch, block_size):
    left_patch = np.reshape(left_patch, (block_size * block_size))
    right_patch = np.reshape(right_patch, (block_size * block_size))
    feature_vector = np.concatenate((left_patch, right_patch), axis=0)

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

    if is_test:
        feature_path = "test_features.npy"
        label_path = "test_labels.npy"

        # save the feature vectors and labels for later use
        np.save(feature_path, feature_vectors)
        np.save(label_path, labels)
    else:
        sets = ["train", "valid"]
        trainX, validX, trainY, validY = train_test_split(feature_vectors, labels, test_size = 0.2)

        print("TrainX : ", trainX.shape)
        print("ValidX : ", validX.shape)
        for set_path in sets:
            feature_path = set_path + "_features.npy"
            label_path = set_path + "_labels.npy"

            # save the feature vectors and labels for later use
            np.save(feature_path, feature_vectors)
            np.save(label_path, labels)

def read_data(data_path):
    data = np.load(data_path, allow_pickle = True)
    return data


def stereoSKlearn(model, left_img, right_img, num_of_disparities, block_size):
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
                    prediction = prediction[0]

                    # if patches belong the same place, calculate the score
                    if prediction == 1:
                        score = np.sum(feature_vector)

                        if score < min_score:
                            min_score = score
                            min_score_disparity = disparity   

            if min_score_disparity != -1:
                new_disp_img[y][x] = min_score_disparity * 8

    return new_disp_img

def stereoOpenCV(model, left_img, right_img, num_of_disparities, block_size):
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

from random import shuffle

block_size = 7
testing = True
model_name = "AdaBoost"
model_base_path = "./models"
if testing:
    # testing part
    num_of_disparities = 16

    base_model = cv2.StereoBM_create(blockSize=block_size, numDisparities=num_of_disparities)

    if model_name == "DecisionTree":
        model_path = os.path.join(model_base_path, "decision_tree_model")
        model = pickle.load(open(model_path, mode="rb"))
    elif model_name == "MultinomialNB":
        model_path = os.path.join(model_base_path, "multinomialnb_model")
        model = pickle.load(open(model_path, mode="rb"))
    elif model_name == "GaussianNB":
        model_path = os.path.join(model_base_path, "gaussiannb_model")
        model = pickle.load(open(model_path, mode="rb"))
    elif model_name == "logistic":
        model_path = os.path.join(model_base_path, "lr_model")
        model = pickle.load(open(model_path, mode="rb"))
    elif model_name == "AdaBoost":
        model_path = os.path.join(model_base_path, "adaboost_model")
        model = pickle.load(open(model_path, mode="rb"))

    test_dataset_path = "test_dataset"
    create_dataset(test_dataset_path, block_size, is_test=True)

    # predict for whole test set to get confusion matrix and other scores
    X = read_data("test_features.npy")
    y = np.float32(read_data("test_labels.npy"))

    # do the prediction
    preds = model.predict(X)

    # calculate the scores
    cm = confusion_matrix(y, preds, labels = [1, -1])
    (prec, recall, f1, _) = precision_recall_fscore_support(y, preds, average="binary", pos_label=1)
    acc = accuracy_score(y, preds)

    print(cm)
    print("Accuracy :", acc, " Precision : ", prec, " Recall : ", recall, " F1 : ", f1)

    dataset_dir = os.listdir(test_dataset_path)
    for directory in dataset_dir:
        img_path = os.path.join(test_dataset_path, directory)

        left_img = cv2.imread(img_path + "/im6.ppm", 0)
        right_img = cv2.imread(img_path + "/im2.ppm", 0)
        disp_img = cv2.imread(img_path + "/disp6.pgm", 0)
        # compute base disparity using block based matching
        # divide by 2 to make disparity calculation similar
        base_disparity_img = base_model.compute(right_img, left_img) / 2

        predicted_disp_img = stereoSKlearn(model, left_img, right_img, num_of_disparities, block_size)

        our_mse = (np.square(disp_img - predicted_disp_img)).mean(axis = None)
        base_mse = (np.square(disp_img - base_disparity_img)).mean(axis = None)
        print("Base MSE : ", base_mse)
        print("Our  MSE : ", our_mse)

        plt.imshow(predicted_disp_img)
        plt.show()

else:
    # training part
    dataset_path = "train_dataset/"
    create_dataset(dataset_path, block_size)

    X = read_data("features.npy")
    y = read_data("labels.npy")

    if model_name == "DecisionTree":
        dt = DecisionTreeClassifier()
        dt.fit(X, y)
        model_path = os.path.join(model_base_path, "decision_tree_model")
        pickle.dump(dt, open(model_path, mode="wb"))
    elif model_name == "MultinomialNB":
        mnb = MultinomialNB()
        mnb.fit(X, y)
        model_path = os.path.join(model_base_path, "multinomialnb_model")
        pickle.dump(mnb, open(model_path, mode="wb"))
    elif model_name == "GaussianNB":
        gnb = GaussianNB()
        gnb.fit(X, y)
        model_path = os.path.join(model_base_path, "gaussiannb_model")
        pickle.dump(gnb, open(model_path, mode="wb"))
    elif model_name == "logistic":
        lr = LogisticRegression(max_iter=100)
        lr.fit(X, y)
        model_path = os.path.join(model_base_path, "lr_model")
        pickle.dump(lr, open(model_path, mode="wb"))
    elif model_name == "AdaBoost":
        ad = AdaBoostClassifier(learning_rate=0.5,n_estimators=5)
        ad.fit(X, y)
        model_path = os.path.join(model_base_path, "adaboost_model")
        pickle.dump(ad, open(model_path, mode="wb"))