import cv2
import stitch
import extractImages
import timeit
import os

video = str(input('Please choose a video from this directory: '))
while not os.path.exists(f"{video}.mp4"):
    video = str(input('Please exist video directory: '))
path = os.path.realpath(f"{video}.mp4")
print(path)
# Extract number of frames in half second difference.
camera = cv2.VideoCapture(path)
# Enter second of start cutting and and (second*2) of ending cutting
extractImages.videoToFrames(camera, 2, 8)
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
