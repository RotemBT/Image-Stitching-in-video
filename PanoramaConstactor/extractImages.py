import cv2
import os


def videoToFrames(cam, startFrame, endFrame):
    """
    Extract Images from requested video, take a frame every half second.
    :param cam:
    :param startFrame:
    :param endFrame:
    :return:
    """
    try:
        # creating a folder named data
        if not os.path.exists('frames'):
            os.mkdir('frames')
        # if not created then raise error
    except OSError:
        print('Error: Creating directory ')

    while (True):
        cam.set(cv2.CAP_PROP_POS_MSEC, (startFrame * 500))
        # reading from frame
        hasFrame, frame = cam.read()

        if hasFrame and startFrame <= endFrame:
            # if video is still left continue creating images
            name = f'./frames/frame' + str(startFrame) + '.jpg'
            print('Creating...' + name)
            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            startFrame += 1
        else:
            break
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


def framesList(mainFolder):
    """
    Warp the images is a list
    :param mainFolder:
    :return:
    """
    folder = os.listdir(mainFolder)
    images = []
    for image in folder:
        curImg = cv2.imread(f'{mainFolder}/{image}')
        # curImg = cv2.resize(curImg, (0, 0), None, 0.2, 0.2)
        images.append(curImg)
    return images
