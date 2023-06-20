import cv2
import numpy as np


class PerspectiveHandler:
    
    @classmethod
    def transform(cls, from_image, to_image, corners):
        h, w, _ = from_image.shape
        points = np.array([[0, 0],  [0, h], 
                           [w, 0],  [w, h]]).astype(np.float32)
        
        matrix = cv2.getPerspectiveTransform(points, corners)
        return cv2.warpPerspective(from_image, matrix, 
                    (to_image.shape[1], to_image.shape[0]))
    
    @classmethod
    def replace_by_mask(cls, from_image, to_image):
        inv_image_mask = [from_image == 0][0]
        to_image[~inv_image_mask] = from_image[~inv_image_mask]
        return to_image