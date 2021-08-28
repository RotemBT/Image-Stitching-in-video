import cv2
import numpy as np
import matplotlib.pyplot as plt


def findAndDescribeFeatures(image):
    """
    find and describe features of image,
    ORB algorithm is used.
    @Return keypoints and features of img.
    """
    # Getting gray image
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=3000)
    # Find interest points and Computing features.
    keypoints, features = orb.detectAndCompute(grayImage, None)
    # Converting keypoints to numbers.
    features = np.float32(features)
    return keypoints, features


def matchFeatures(featuresA, featuresB, ratio=0.75):
    """
    Matching features between 2 features.
    If opt='BF', BruteForce algorithm is used.
    @ratio is the Lowe's ratio test.
    @return matches
    """
    featureMatcher = cv2.DescriptorMatcher_create("BruteForce")
    # Performs KNN matching between the two feature vector sets using k=2
    # (indicating the top two matches for each feature vector are returned).
    matches = featureMatcher.knnMatch(featuresA, featuresB, k=2)

    # store all the bestMatches matches as per Lowe's ratio test.
    bestMatches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            bestMatches.append(m)
    if len(bestMatches) > 4:
        return bestMatches
    raise Exception("Not enough matches")


def generateHomography(sourceImage, destinationImage, ransacRep=5.0):
    """
    @Return Homography matrix,
    @param sourceImage is the image which is warped by homography,
    @param destinationImage is the image which is choosing as pivot,
    @param ransacRep is the maximum pixel “wiggle room” allowed by the RANSAC algorithm
    """
    # Find keypoints and descriptors of the images.
    sourceKeypoints, sourceFeatures = findAndDescribeFeatures(sourceImage)
    destinationKeyPoints, destinationFeatures = findAndDescribeFeatures(destinationImage)
    # Find the best matches between the keypoints.
    bestMatches = matchFeatures(sourceFeatures, destinationFeatures)
    # Convert keypoints to an argument for findHomography.
    # match.queryIdx give the index of the features in the list of query features
    # the list of the query features is of the image we would like to spread
    # the others is the training features for training, this is the routine before homography
    sourcePoints = np.float32([sourceKeypoints[m.queryIdx].pt for m in bestMatches]).reshape(-1, 1, 2)
    destinationPoints = np.float32([destinationKeyPoints[m.trainIdx].pt for m in bestMatches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, ransacRep)
    # plt.figure()
    # plt.imshow(drawKeypoints(sourceImage, sourceKeypoints)[:,:,::-1])
    # plt.imshow(drawMatches(sourceImage, sourceKeypoints, destinationImage, destinationKeyPoints, bestMatches, mask)
    # [:,:,::-1])
    # plt.show()
    matchesMask = mask.ravel().tolist()
    return homography, matchesMask


def drawKeypoints(img, keypoints):
    return cv2.drawKeypoints(img, keypoints, None, flags=None)


def drawMatches(sourceImage, sourceKeypoints, destinationImage, destinationKeypoints, matches, matchesMask):
    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask[:100],  # draw only inlines
        flags=2,
    )
    return cv2.drawMatches(
        sourceImage, sourceKeypoints, destinationImage, destinationKeypoints, matches[:100], None, **draw_params)
