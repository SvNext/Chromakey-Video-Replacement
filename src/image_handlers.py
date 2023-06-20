import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image, ImageFilter, ImageOps, \
    ImageEnhance

from pathlib import Path


class ImageCVHandler:
    
    @classmethod
    def load_image(cls, dirname: str, filename: str):
        path = Path(dirname).joinpath(filename)
        if not Path.exists(path):
            raise Exception("FileNotFoundError")
        return cv2.imread(str(path))
            
    @classmethod
    def plot_image(cls, image: np.ndarray, DPI: int = 30):
        h, w  = image.shape[:2]
        fig, ax = plt.subplots(1, 1, figsize = (w / float(DPI), h / float(DPI)))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        ax.imshow(image, cmap='gray')
        plt.show()
        
    @classmethod
    def convert_BGR_to_RGB(cls, image: np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @classmethod
    def convert_RGB_to_HSV(cls, image: np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
    @classmethod
    def convert_BGR_to_HSV(cls, image: np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    @classmethod
    def convert_BGR_to_GRAY(cls, image: np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class ImagePILHandler:
    
    @classmethod
    def improve_image(cls, image: Image):
        
        image_filtered = image.filter(ImageFilter.ModeFilter(size = 5))
        image_filtered = image_filtered.filter(ImageFilter.MedianFilter(size = 3))
        image_filtered = image_filtered.filter(ImageFilter.GaussianBlur(radius = 2))
        return image_filtered