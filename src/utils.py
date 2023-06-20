import cv2
import numpy as np


def resize_image(image, scale: float = 2):
    w = int(image.shape[1] * scale)
    h = int(image.shape[0] * scale)
    resized = cv2.resize(image, (w, h), 
                         interpolation = cv2.INTER_LINEAR)
    return resized


def replace_area(image, area, crop_borders):
    image[crop_borders['w'][0]:crop_borders['w'][1], 
          crop_borders['h'][0]:crop_borders['h'][1]] = area
    return image