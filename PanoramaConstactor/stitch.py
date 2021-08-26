import cv2
import numpy as np
import features


def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
    """
    Make a mask of required height width and barrier.
    :param height:
    :param width:
    :param barrier:
    :param smoothing_window:
    :param left_biased:
    :return:
    """
    assert barrier < width
    # Make a emtpy mask with required size
    mask = np.zeros((height, width))
    # Fill the mask
    offset = int(smoothing_window / 2)
    try:
        if left_biased:
            mask[:, barrier - offset: barrier + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset + 1).T, (height, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset: barrier + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset + 1).T, (height, 1)
            )
            mask[:, barrier + offset:] = 1
    except BaseException:
        if left_biased:
            mask[:, barrier - offset: barrier + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (height, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset: barrier + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (height, 1)
            )
            mask[:, barrier + offset:] = 1

    return cv2.merge([mask, mask, mask])


def panoramaBlending(destinationImage, sourceImage, widthDestination, side, showStep=False):
    """
    Given two aligned images, and the @width_dst is width of dst_img
    before resize, that indicates where there is the discontinuity between the images,
    this function produce a smoothed transient in the overlapping.
    @smoothing_window is a parameter that determines the width of the transient
    left_biased is a flag that determines whether it is masked the left image,
    or the right one
    """

    height, width, _ = destinationImage.shape
    smoothing_window = int(widthDestination / 8)
    barrier = widthDestination - int(smoothing_window / 2)
    mask1 = blendingMask(height, width, barrier, smoothing_window=smoothing_window, left_biased=True)
    mask2 = blendingMask(height, width, barrier, smoothing_window=smoothing_window, left_biased=False)

    if showStep:
        nonBlend = sourceImage + destinationImage
    else:
        nonBlend = None
        leftSide = None
        rightSide = None

    if side == "left":
        destinationImage = cv2.flip(destinationImage, 1)
        sourceImage = cv2.flip(sourceImage, 1)
        destinationImage = destinationImage * mask1
        sourceImage = sourceImage * mask2
        panorama = sourceImage + destinationImage
        panorama = cv2.flip(panorama, 1)
        if showStep:
            leftSide = cv2.flip(sourceImage, 1)
            rightSide = cv2.flip(destinationImage, 1)
    else:
        destinationImage = destinationImage * mask1
        sourceImage = sourceImage * mask2
        panorama = sourceImage + destinationImage
        if showStep:
            leftSide = destinationImage
            rightSide = sourceImage

    return panorama, nonBlend, leftSide, rightSide


def warpTwoImages(sourceImage, destinationImage, showStep=False):
    """
    Wrap two images with a suitable homography.
    :param sourceImage: 
    :param destinationImage: 
    :param showStep:
    :return: 
    """
    # Generate Homography matrix.
    homography, _ = features.generateHomography(sourceImage, destinationImage)

    # Get height and width of the two images.
    heightSource, widthSource = sourceImage.shape[:2]
    heightDestination, widthDestination = destinationImage.shape[:2]

    # When we have established a homography we need to warp perspective
    # Change field of view
    listOfPointsSource = \
        np.float32([[0, 0], [0, heightSource], [widthSource, heightSource], [widthSource, 0]]).reshape(-1, 1, 2)
    listOfPointsDestination = \
        np.float32([[0, 0], [0, heightDestination], [widthDestination, heightDestination], [widthDestination, 0]]) \
            .reshape(-1, 1, 2)

    try:
        # In Perspective Transformation, , we can change the perspective of a given image or video for getting better
        # insights about the required information.
        # In Perspective Transformation, we need provide the points on the image from which want
        # to gather information by changing the perspective.
        # We also need to provide the points inside which we want to display our image.
        # Then, we get the perspective transform from the two given set of points and wrap it with the
        # original image.
        # https://www.geeksforgeeks.org/perspective-transformation-python-opencv/

        # Apply homography to corners of source image
        tempPoints = cv2.perspectiveTransform(listOfPointsSource, homography)
        listOfPoints = np.concatenate((tempPoints, listOfPointsDestination), axis=0)

        # Ravel is a np function that takes a 2 or more dim array and changes it to flatted array.
        # Find max min of x,y coordinate
        [xMin, yMin] = np.int64(listOfPoints.min(axis=0).ravel() - 0.5)
        [_, yMax] = np.int64(listOfPoints.max(axis=0).ravel() + 0.5)
        translationDistance = [-xMin, -yMin]

        # Top left point of image which apply homography matrix, which has x coordinate < 0, so source image
        # merge to left side otherwise merge to right side of the destination image.
        if listOfPoints[0][0][0] < 0:
            side = "left"
            widthPanorama = widthDestination + translationDistance[0]
        else:
            widthPanorama = int(tempPoints[3][0][0])
            side = "right"
        heightPanorama = yMax - yMin

        # Translation
        # https://stackoverflow.com/a/20355545
        homographyMatrix = np.array([[1, 0, translationDistance[0]],
                                     [0, 1, translationDistance[1]],
                                     [0, 0, 1]])
        sourceImageWarped = cv2.warpPerspective(
            sourceImage, homographyMatrix.dot(homography), (widthPanorama, heightPanorama))
        # Generating size of destination image which has the same size as sourceImageWarped.
        destinationImageResized = np.zeros((heightPanorama, widthPanorama, 3))
        if side == "left":
            destinationImageResized[translationDistance[1]: heightSource + translationDistance[1],
            translationDistance[0]: widthDestination + translationDistance[0]] = destinationImage
        else:
            destinationImageResized[translationDistance[1]: heightSource + translationDistance[1],
            :widthDestination] = destinationImage

        # Blending panorama
        panorama, nonBlend, leftSide, rightSide = panoramaBlending(
            destinationImageResized, sourceImageWarped, widthDestination, side, showStep=showStep)

        # Cropping black space
        panorama = crop(panorama, heightDestination, listOfPoints)
        return panorama, nonBlend, leftSide, rightSide
    except BaseException:
        raise Exception("Please try again with another image set!")


def multiStitching(listImages):
    """
    Assume that the listImages was supplied in left-to-right order, choose middle image then
    divide the list into 2 sub-lists, left-list and right-list. Stitching middle image with each
    image in 2 sub-lists. @param listImages is The list which containing images, @param smoothing_window is
    the value of smoothie side after stitched, @param output is the folder which containing stitched image
    """
    size = int(len(listImages) / 2 + 0.5)
    left = listImages[:size]
    right = listImages[size - 1:]
    right.reverse()
    while len(left) > 1:
        destinationImage = left.pop()
        sourceImage = left.pop()
        leftPanorama, _, _, _ = warpTwoImages(sourceImage, destinationImage)
        leftPanorama = leftPanorama.astype('uint8')
        left.append(leftPanorama)

    while len(right) > 1:
        destinationImage = right.pop()
        sourceImage = right.pop()
        rightPanorama, _, _, _ = warpTwoImages(sourceImage, destinationImage)
        rightPanorama = rightPanorama.astype('uint8')
        right.append(rightPanorama)

    # Check if the right panorama width is bigger than left panorama width, select right panorama
    # as a destination otherwise select the left panorama
    if rightPanorama.shape[1] >= leftPanorama.shape[1]:
        completePanorama, _, _, _ = warpTwoImages(leftPanorama, rightPanorama)
    else:
        completePanorama, _, _, _ = warpTwoImages(rightPanorama, leftPanorama)
    return completePanorama


def crop(panorama, heightDestination, corners):
    """
    crop panorama based on destination.
    @param panorama is the panorama
    @param heightDestination is the height of destination image
    @param corners is the tuple which containing 4 corners of warped image and
    4 corners of destination image
    """
    # find max min of x,y coordinate
    [xMin, yMin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    t = [-xMin, -yMin]
    corners = corners.astype(int)

    # corners[0][0][0] is the X coordinate of top-left point of warped image
    # If it has value<0, warp image is merged to the left side of destination image
    # otherwise is merged to the right side of destination image
    if corners[0][0][0] < 0:
        n = abs(-corners[1][0][0] + corners[0][0][0])
        panorama = panorama[t[1]: heightDestination + t[1], n:, :]
    else:
        if corners[2][0][0] < corners[3][0][0]:
            panorama = panorama[t[1]: heightDestination + t[1], 0: corners[2][0][0], :]
        else:
            panorama = panorama[t[1]: heightDestination + t[1], 0: corners[3][0][0], :]
    return panorama
