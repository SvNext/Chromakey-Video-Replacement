import cv2
import numpy as np


class MaskHandler:
    
    lower_green = (40, 100,  40)
    upper_green = (80, 255, 255)
    
    @classmethod
    def select_green_zone(cls, image: np.ndarray):
        return cv2.inRange(image, cls.lower_green, cls.upper_green)
    
    
    @classmethod
    def blure_mask(cls, mask: np.ndarray):
        kernel = np.ones((3,3),np.uint8)
        # mask_blure = cv2.GaussianBlur(mask.copy(), (3, 3), 0)
        mask_blure = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_blure = cv2.morphologyEx(mask_blure, cv2.MORPH_CLOSE, kernel)
        return mask_blure
    
    @classmethod
    def findout_gradient(cls, mask: np.ndarray):
        kernel = np.ones((3,3),np.uint8)
        gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        return gradient