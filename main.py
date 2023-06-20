import numpy as np

import cv2
from PIL import Image

from pathlib import Path
from loguru import logger

from src import resize_image, replace_area
from src import WindowInMoment, WindowsContainer
from src import ImageCVHandler, ImagePILHandler, MaskHandler, \
    BorderHandler, PerspectiveHandler


class Pipeline:
    
    def __init__(self, wcontainer: WindowsContainer):
        self._wcontainer = wcontainer
        
    
    def find_windows_region(self, image: np.array):
        img_hsv = ImageCVHandler.convert_BGR_to_HSV(image)
        
        mask = MaskHandler.select_green_zone(img_hsv)
        gradient = MaskHandler.findout_gradient(mask)
        
        contours = BorderHandler.findout_contours(gradient.copy())
        contours = BorderHandler.filter_contours(contours)
        
        croped_image, windows_borders = BorderHandler.crop_image(image, np.vstack(contours))
        return croped_image, windows_borders
        
    def find_windows_borders(self, image: np.array):
        img_rgb = ImageCVHandler.convert_BGR_to_RGB(image)
        img_rgb = Image.fromarray(img_rgb, mode = 'RGB')
        
        improved_img = ImagePILHandler.improve_image(img_rgb)
        img_hsv = ImageCVHandler.convert_RGB_to_HSV(np.array(improved_img))
        
        mask = MaskHandler.select_green_zone(img_hsv)
        mask = MaskHandler.blure_mask(mask.copy())
        gradient = MaskHandler.findout_gradient(mask)
        
        contours = BorderHandler.findout_contours(gradient.copy())
        contours = BorderHandler.filter_contours(contours )
        polygons = BorderHandler.findout_polygons(contours)
        
        return {
            'mask': mask, 
            'gradient': gradient,
            'contours': contours, 
            'polygons': polygons,
            'improved_img': np.array(improved_img)
        }
    
    
    def save_windows_data(self, windows_data: dict):
        items = []
        for i, (contour, polygon) in enumerate(zip(windows_data['contours'], 
                                                   windows_data['polygons'])):
                
            centroid = BorderHandler.findout_window_centroid(contour)
            geometry = BorderHandler.findout_window_area_perimeter(contour)
            corners  = BorderHandler.findout_corners(polygon, centroid)
            
            item = WindowInMoment(
                geometry['area'], geometry['perimeter'], 
                centroid['cX'], centroid['cY'], *corners            
            )
        
#             item = self._wcontainer.append_item_simple(item)
#             items.append(item)
            
            updated_item = self._wcontainer.append_item_ema(item)
            items.append(updated_item)
        
        return items
    
    
    def write_video(self, video_path, output_path, resume_image):
        video = cv2.VideoCapture(video_path)

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH ))
        frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        frame_num = 0
        
        while True:
            frame_num += 1
            ret, frame = video.read()

            if not ret:
                break
            
            # select active zone
            croped_image, windows_borders = self.find_windows_region(frame.copy())
            croped_image = resize_image(croped_image.copy(), 2)
        
            windows_data = self.find_windows_borders(croped_image.copy())
            zoomed_image = cv2.bitwise_and(windows_data['improved_img'], 
                                           windows_data['improved_img'], mask=~windows_data['gradient'])
            # plot corners
            for polygon in windows_data['polygons']:
                polygon = polygon.reshape(polygon.shape[0], -1)
                
                for point in polygon:
                    zoomed_image = cv2.circle(zoomed_image, point, radius=3, 
                                           color=(0, 0, 255), thickness=-1)
            
            # make perspective transformations
            new_croped_image = croped_image.copy()
            for item in self.save_windows_data(windows_data):
                rotated_image = PerspectiveHandler.transform(resume_image, 
                                            croped_image, item.corners)
                
                new_croped_image = PerspectiveHandler.replace_by_mask(rotated_image, new_croped_image)
            
            # invert resize
            new_croped_image = resize_image(new_croped_image.copy(), 0.5)
            new_frame = replace_area(frame.copy(), new_croped_image, windows_borders)
                     
            logger.info(f'Frame Num: {frame_num}')
            output.write(new_frame)
            if cv2.waitKey(1) == 27:
                break

        video.release()
        output.release()
        cv2.destroyAllWindows()
        
        
    def write_mask(self, video_path, output_path, resume_image):
        video = cv2.VideoCapture(video_path)

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH ))
        frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        frame_num = 0
        
        while True:
            frame_num += 1
            ret, frame = video.read()

            if not ret:
                break
            
            # select active zone of image
            croped_image, windows_borders = self.find_windows_region(frame.copy())
            croped_image = resize_image(croped_image.copy(), 2)
        
            windows_data = self.find_windows_borders(croped_image.copy())
            zoomed_image = cv2.bitwise_and(windows_data['improved_img'], 
                                           windows_data['improved_img'], mask=~windows_data['gradient'])
            # plot corners
            for polygon in windows_data['polygons']:
                polygon = polygon.reshape(polygon.shape[0], -1)
                
                for point in polygon:
                    zoomed_image = cv2.circle(zoomed_image, point, radius=3, 
                                           color=(0, 0, 255), thickness=-1)
    
            # add zoomed_image to original image
            step = 200
            replace_coord = {
                'w': [step, zoomed_image.shape[0] + step], 
                'h': [step, zoomed_image.shape[1] + step]
            }        
            
            new_frame = replace_area(frame, zoomed_image, replace_coord)            
            logger.info(f'Frame Num: {frame_num}')
            
            output.write(new_frame)
            if cv2.waitKey(1) == 27:
                break

        video.release()
        output.release()
        cv2.destroyAllWindows()
        


if __name__ == '__main__':

    # Usage:
    resume_image = ImageCVHandler.load_image(
        './data', 'resume_img.png')


    container = WindowsContainer(**{
        'num_windows': 2,
        'window': 7, 'gamma': 0.6
    })

    pipeline = Pipeline(container)
    pipeline.write_video('data/input_video.mp4', 
                         'data/output_video.mp4', resume_image)


    ##### MAKE ZOOMED VERSION OF VIDEO #####
    # container = WindowsContainer(**{
    #     'num_windows': 2,
    #     'window': 7, 'gamma': 0.6
    # })
    #pipeline = Pipeline(container)
    #pipeline.write_mask('data/input_video.mp4', 'data/mask_video.mp4', resume_image)