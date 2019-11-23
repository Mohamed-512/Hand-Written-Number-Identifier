from PIL import Image
import numpy as np

"""
This is a small script to help in every 
image processing task need to be done on the fly
"""


def split_colors(arr):
    """
    Split colors of a given grey scale image into black and white only based on a threshold
    to be changes in different lighting conditions. threshold is recommended to be 3 in dim
    lighting and 4 in highly illuminated lighting.
    :param arr: grey scale image to be modified 2D numpy array
    :return: Modified image
    """
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            # if pixel value is very white turn it to black
            if arr[i, j] < 255//2.5:
                arr[i, j] = 255
            # Otherwise turn it to white
            else:
                arr[i, j] = 0
    return arr


def make_square_by_padding(uneven_arr):
    """
    Coverts a rectangular grey scale image to square image by adding padding
    :param uneven_arr: Rectangular shaped image. 2D numpy array
    :return: Squared image after adding padding
    """
    # Height < Width
    if uneven_arr.shape[0] < uneven_arr.shape[1]:
        zeros = np.zeros((uneven_arr.shape[1], (uneven_arr.shape[1] - uneven_arr.shape[0]) // 2))
        try:
            # Add empty pixels to the top and bottom
            np.vstack((zeros, uneven_arr, zeros))
        except:
            pass
    # Height > Width
    elif uneven_arr.shape[0] > uneven_arr.shape[1]:
        # Height > Width
        # Add empty pixels to the right and left
        zeros = np.zeros((uneven_arr.shape[0], (uneven_arr.shape[0] - uneven_arr.shape[1]) // 2))
        uneven_arr = np.hstack((zeros, uneven_arr, zeros))

    return uneven_arr


def down_sample(arr, size=(28, 28)):
    """
    downsizes an image
    :param arr: given grey scale image to be resized. 2D numpy array
    :param size: new size (width, height). by default set to (28, 28) same as MNIST data set
    :return: Downsized version of the image accrding to the new size
    """
    arr = np.array(Image.fromarray(arr).resize(size))
    return arr


