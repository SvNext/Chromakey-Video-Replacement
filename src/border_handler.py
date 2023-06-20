import cv2
import numpy as np


class BorderHandler:
    
    @classmethod
    def filter_contours(cls, contours: tuple, num_cnts: int = 2):
        cnts = [x for x in contours if x.shape[0] > 3]
        
        areas = [cv2.contourArea(c) for c in cnts]
        (cnts, _) = zip(*sorted(zip(cnts, areas),
                        key=lambda x: x[1], reverse=True))
        return cnts[:num_cnts]
    
    @classmethod
    def findout_contours(cls, mask: np.ndarray):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
        return cnts
    
    @classmethod
    def findout_polygons(cls, contours):
        polygons = []
        for contour in contours:
            epsilon = 0.1 * cv2.arcLength(contour, True)
            polygons.append(cv2.approxPolyDP(contour, epsilon, True))
        
        return polygons
    
    @classmethod
    def findout_corners(cls, polygon: np.ndarray, centroid: dict):    
        polygon = polygon.reshape(polygon.shape[0], -1)
    
        for point in polygon:
            if (point[0] < centroid['cX']) and (point[1] < centroid['cY']): corner_1 = point
            elif (point[0] < centroid['cX']) and (point[1] > centroid['cY']): corner_2 = point
            elif (point[0] > centroid['cX']) and (point[1] < centroid['cY']): corner_3 = point
            elif (point[0] > centroid['cX']) and (point[1] > centroid['cY']): corner_4 = point

        return np.array([corner_1, corner_2, 
                         corner_3, corner_4]).astype(np.int32)
    
    @classmethod
    def findout_window_centroid(cls, contour: np.ndarray):
        M = cv2.moments(contour)
        return {
            'cX': int(M['m10'] / M['m00']), 
            'cY': int(M['m01'] / M['m00'])
        }
    
    @classmethod
    def findout_window_area_perimeter(cls, contour: np.ndarray):
        return {
            'area': cv2.contourArea(contour),
            'perimeter': cv2.arcLength(contour, True)
        }
    
    
    @classmethod
    def crop_image(cls, image, contour, padding_px: int = 10):
        borders = contour.reshape(contour.shape[0], -1)
        x_min, x_max = np.min(borders[:, 0]) - padding_px, np.max(borders[:, 0]) + padding_px
        y_min, y_max = np.min(borders[:, 1]) - padding_px, np.max(borders[:, 1]) + padding_px
        croped_image = image[y_min:y_max, x_min:x_max, :]
        return croped_image, {
            'h': [x_min, x_max],
            'w': [y_min, y_max],
        }