import cv2
import stitch
import extractImages
import timeit
import os

"""
1.Import 2 images
2.convert to gray scale
3.Initiate ORB detector
4.Find key points and describe them
5.Match keypoints- Brute force matcher
6.RANSAC(reject bad keypoints)
7.Register two images (use homography) 
"""
video = str(input('Please choose a video from this directory: '))
while not os.path.exists(f"{video}.mp4"):
    video = str(input('Please exist video directory: '))
path = os.path.realpath(f"{video}.mp4")
print(path)
# Extract number of frames in half second difference.
camera = cv2.VideoCapture(path)
extractImages.videoToFrames(camera, 0, 5)
# Calculate execution time
print("Processing....")
start = timeit.default_timer()
# Load images from video
list_images = extractImages.framesList('frames')

# Create panorama, using ORB algorithm.
panorama = stitch.multiStitching(list_images)

# Save the panorama image
cv2.imwrite("panorama.jpg", panorama)

stop = timeit.default_timer()
print("Complete!")
print("Execution time: ", stop - start)
# Clean the images

for image in os.listdir('frames'):
    os.remove(os.path.realpath('frames') + '/' + image)
