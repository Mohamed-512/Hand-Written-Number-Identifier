import cv2
import numpy as np
import tensorflow.keras.models as models
import ImageProcessor

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

camera = PiCamera()
camera.resolution = (480, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(480, 480))
model = models.load_model('mnist_model.h5')
arr_map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]


def test(image):
    """
    :param image: Image to test the model on
    :return: 1D array of prediction each index representing a vote to a specific number
    """
    p = model.predict(np.array([image]))
    return p


def get_top_stat(full_output):
    """
    * USED FOR DEBUGGING
    :param full_output:  1D array of prediction each index representing a vote to a specific number
    :return: Dictionary containing top 3 voted numbers and their votes
    """
    # Conversion to python list
    full_output = list(full_output)

    # Variable initialization
    output = {}

    for i in range(3):
        max_num = max(full_output)
        # Choosing top 3 as long it has a vote higher than 0.1%
        if max_num < 0.001:
            return output
        max_index = full_output.index(max_num)
        full_output.remove(max_num)
        output.update({str(max_index): str(max_num)})

    return output


def draw_label(img, text, pos, font_color, bg_color, font_size):
    """
    * USED FOR DEBUGGING
    Writes a text on a specific position of a circle
    :param img: image to write on
    :param text: text to write
    :param pos: (x, y) coordinates where writing should start
    :param font_color: (R, G, B) color
    :param bg_color: (R, G, B) color
    :param font_size: size (int)
    :return: None
    """
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = cv2.FILLED
    margin = 0

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, font_color, font_size, cv2.LINE_AA)


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    frame = frame.array

    #################################   DRAWING BOXes #################################

    # Converting frame to grey scale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applying gaussian blur on frame to decrease noise
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Inverting grey scale frame to Black and white based on some threshold to enhance edge detection
    _, im_th = cv2.threshold(frame, 127, 250, cv2.THRESH_BINARY_INV)

    # Generating contours from the thresholded image *not frame*
    contours, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Getting rectangles from contours
    rectangles = [cv2.boundingRect(ctr) for ctr in contours]

    # Iterating through each generated rectangle
    for rect in rectangles:
        # Extracting (x, y) coordinates, width, and height from the rectangle
        x, y, width, height = rect

        # Calculating area
        area = width * height

        # If area is so tiny, too big, very narrow, or very short its neglected, not processed, and not drawn
        if area < 300 or area > 10000 or width < 20 or height < 20:
            continue

        # Extract the pixels from frame within box bounds by reference not by value to be edited and make
        # the edit appear in the final frame
        cropped_img = frame[y:y + height, x:x + width]
        # For those pixels of the box the colors are heavily split to Black and White only
        cropped_img = ImageProcessor.split_colors(cropped_img)

        only_cropped = np.array(cropped_img)
        # Applying padding to the image
        only_cropped = ImageProcessor.make_square_by_padding(only_cropped)

        # Down sampling the image
        down_sample_img = np.array(ImageProcessor.down_sample(only_cropped))

        # Adding extra dimension to meet the model's need and MNIST data fomat
        down_sample_img = np.expand_dims(down_sample_img, axis=3)

        # Predicting the votes for all numbers based on the final modified image
        all_pred = test(down_sample_img)[0]
        # Getting the index of the highest vote
        prediction = np.argmax(all_pred)

        # Drawing a rectangle around the currently working with image in the original frame capture
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
        # Writing prediction above the current rectangle in the original frame capture
        cv2.putText(frame, str(prediction), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)

    ##################################################################################

    cv2.imshow('frame0', frame)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
        break
