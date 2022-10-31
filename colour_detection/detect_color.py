import math
import operator
import os

import cv2
import numpy as np

training_feature_vector = []  # training feature vector


# calculation of euclidead distance
def calculate_euclidean_distance(variable1, variable2, length):
    distance = 0
    for x in range(length):
        distance += pow(variable1[x] - variable2[x], 2)
    return math.sqrt(distance)


# get k nearest neigbors
def k_nearest_neighbors(test_instance, k):
    distances = []
    length = len(test_instance)
    for x in range(len(training_feature_vector)):
        dist = calculate_euclidean_distance(
            test_instance, training_feature_vector[x], length
        )

        distances.append((training_feature_vector[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# votes of neighbors
def response_of_neighbors(neighbors):
    all_possible_neighbors = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in all_possible_neighbors:
            all_possible_neighbors[response] += 1
        else:
            all_possible_neighbors[response] = 1
    sortedVotes = sorted(
        all_possible_neighbors.items(), key=operator.itemgetter(1), reverse=True
    )
    return sortedVotes[0][0]


def color_histogram_of_image(image):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    features = []
    feature_data = []
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = elem
        elif counter == 2:
            green = elem
        elif counter == 3:
            red = elem

    feature_data.append(red)
    feature_data.append(green)
    feature_data.append(blue)
    return feature_data


def color_histogram_of_training_image(img_name):
    # detect image color by using image file name to label training data
    if "red" in img_name:
        data_source = "red"
    elif "yellow" in img_name:
        data_source = "yellow"
    elif "green" in img_name:
        data_source = "green"
    elif "orange" in img_name:
        data_source = "orange"
    elif "white" in img_name:
        data_source = "white"
    elif "black" in img_name:
        data_source = "black"
    elif "blue" in img_name:
        data_source = "blue"
    elif "violet" in img_name:
        data_source = "violet"

    # load the image
    image = cv2.imread(img_name)
    feature_data = color_histogram_of_image(image)
    feature_data.append(data_source)

    training_feature_vector.append(feature_data)


def training():
    # red color training images
    for f in os.listdir("colour_detection/training_dataset/red"):
        color_histogram_of_training_image("colour_detection/training_dataset/red/" + f)

    # yellow color training images
    for f in os.listdir("colour_detection/training_dataset/yellow"):
        color_histogram_of_training_image("colour_detection/training_dataset/yellow/" + f)

    # green color training images
    for f in os.listdir("colour_detection/training_dataset/green"):
        color_histogram_of_training_image("colour_detection/training_dataset/green/" + f)

    # orange color training images
    for f in os.listdir("colour_detection/training_dataset/orange"):
        color_histogram_of_training_image("colour_detection/training_dataset/orange/" + f)

    # white color training images
    for f in os.listdir("colour_detection/training_dataset/white"):
        color_histogram_of_training_image("colour_detection/training_dataset/white/" + f)

    # black color training images
    for f in os.listdir("colour_detection/training_dataset/black"):
        color_histogram_of_training_image("colour_detection/training_dataset/black/" + f)

    # blue color training images
    for f in os.listdir("colour_detection/training_dataset/blue"):
        color_histogram_of_training_image("colour_detection/training_dataset/blue/" + f)


def main(image):
    test_feature_vector = color_histogram_of_image(image)
    classifier_prediction = []  # predictions
    k = 3  # K value of k nearest neighbor

    for x in range(len(test_feature_vector)):
        neighbors = k_nearest_neighbors(test_feature_vector, k)
        result = response_of_neighbors(neighbors)
        classifier_prediction.append(result)
    return classifier_prediction[0]


def detect_color(box_image):
    # read the test image
    source_image = box_image
    prediction = "n.a."

    if training_feature_vector:
        # get the prediction
        prediction = main(source_image)
    else:
        training()
        # get the prediction
        prediction = main(source_image)

    return prediction
