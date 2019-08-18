import argparse
import copy
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--show_image', type=str, default='no', help='yes or no')

DEFAULT_CAMERA_PORT = 0

COLORS = ('red', 'green', 'blue')


if __name__ == '__main__':
    args = parser.parse_args()
    camera = cv2.VideoCapture(DEFAULT_CAMERA_PORT)
    camera.set(cv2.CAP_PROP_FPS, 1)
    camera.set(cv2.CAP_PROP_BRIGHTNESS, 200)
    # Wait for parameters to set.
    time.sleep(.1)

    success, img = camera.read()

    plt.imsave('temp.jpg', img)

    for i in range(3):
        img_monochrome = np.zeros(img.shape, dtype='uint8')
        img_monochrome[:, :, i] = img[:, :, i]
        plt.imsave(f'temp_{COLORS[i]}.jpg', img_monochrome)

    if args.show_image == 'yes':
        plt.imshow(img)
        plt.show()
