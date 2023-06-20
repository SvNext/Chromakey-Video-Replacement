import copy
import numpy as np

import cv2
from loguru import logger
from dataclasses import dataclass



@dataclass
class WindowInMoment:

    area: float
    perimeter: int
    cX: int
    cY: int
    corner_1: np.ndarray
    corner_2: np.ndarray
    corner_3: np.ndarray
    corner_4: np.ndarray
    
    def __init__(self, 
            area: float, perimeter: int, cX: int, cY: int,
            corner_1: np.ndarray, corner_2: np.ndarray,
            corner_3: np.ndarray, corner_4: np.ndarray,
        ):
        
        self.area = area
        self.perimeter = perimeter
        self.cX = cX
        self.cY = cY
        self.corner_1 = corner_1
        self.corner_2 = corner_2
        self.corner_3 = corner_3
        self.corner_4 = corner_4
        
    @property
    def corners(self):
        return np.array([self.corner_1, self.corner_2, 
                         self.corner_3, self.corner_4]).astype(np.float32)

class WindowsContainer:
    
    def __init__(self, num_windows:int = 2, 
            window: int = 20, gamma: float = 0.4
        ):
        self.window = window
        self.gamma = gamma
        
        self.relative   = { k: [] for k in range(num_windows)}
        self._container = { k: [] for k in range(num_windows)}
        
    def findout_key(self, item2: WindowInMoment):
        distances = []
        for k in self._container:
            if len(self._container[k]) != 0:
                item1 = self._container[k][-1]
                dist = (item1.cX - item2.cX) ** 2 + (item1.cY - item2.cY) ** 2
                distances.append((k, dist))
            else:
                distances.append((k, 0))
            
        distances = sorted(distances, key = lambda x: x[1])
        return distances[0][0]
        
        
    def append_item_simple(self, item2: WindowInMoment):
        key = self.findout_key(item2)
        self._container[key].append(item2)
        return item2
        
    @property
    def get_window_keys(self):
        return list(self._container.keys())
    
    def get_area_values(self, key: int) -> list:
        return [x.area for x in self._container[key]]
    
    def get_perimeter_values(self, key: int) -> list:
        return [x.perimeter for x in self._container[key]]
    
    def get_centroid_values(self, key: int) -> np.array:
        return np.array([[x.cX, x.cY] for x in self._container[key]])
    
    def get_corner_values(self, key: int) -> np.array:
        return np.array([[x.corner_1, x.corner_2, x.corner_3, x.corner_4] 
                         for x in self._container[key]])
    
    def adapt_params(self, item: WindowInMoment, key: int, 
            eps_max: float = 0.02, eps_min: float = 0.01
        ):
        
        perimeters = self.get_perimeter_values(key)
        relative_difference = np.abs(1 -  np.mean(perimeters[-self.window :]) / item.perimeter)
        self.relative[key].append(relative_difference)
        if relative_difference > eps_max:
            return 1, 0.95
        elif eps_min < relative_difference < eps_max:
            return 3, 0.85
        
        return self.window, self.gamma
    
    
    def append_item_ema(self, item2: WindowInMoment):
        key = self.findout_key(item2)
        prev_corners = self.get_corner_values(key)
        
        if len(prev_corners):
            window, gamma = self.adapt_params(item2, key)
            corners_ema = prev_corners[-window:, :].mean(axis = 0) * (1 - gamma) + gamma * item2.corners

            new_item = copy.copy(item2)
            new_item.corner_1 = corners_ema[0]
            new_item.corner_2 = corners_ema[1]
            new_item.corner_3 = corners_ema[2]
            new_item.corner_4 = corners_ema[3]

            self._container[key].append(new_item)
            return new_item
        
        self._container[key].append(item2)
        return item2