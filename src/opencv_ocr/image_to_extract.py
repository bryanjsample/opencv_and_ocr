'''
    module.py
    Author : Bryan Sample

    DESCRIPTION
'''
import cv2 as cv
from cv2.typing import MatLike
from image_to_process import ImageToProcess
import pytesseract
from pytesseract import Output
from typing import List, Tuple, Dict, Any
import numpy as np
from decorators import add_transformation

class ImageToExtract(ImageToProcess):
    ''''''
    def __init__(self, image_path: str) -> None:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
        super().__init__(image_path)

    @add_transformation('draw contours')
    def draw_contours(self, min_threshold:int=1_000, max_threshold:int=500_000, contour_method:str='tree', contour_mode:str='simple', filter:bool=True,) -> List[List[int]]:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                - min_threshold: minimum area of a contour to be included
                - max_threshold :maximum area of a contour to be include
                - contour_method : string correlating to a key in _methods dictionary
                    - 'external' : cv.RETR_EXTERNAL | retrieves only external colors, disregarding any contours inside the objects
                    - 'list' : cv.RETR_LIST | retrives all contours without any hierarchy
                    - 'ccomp' : cv.RETR_CCOMP | retrieves all contours and organizes them into a two-level hierarchy. Top level contains the outer boundaries of the objects, and the second level contains the boundaries of the inner holes
                    - 'tree' : cv.RETR_TREE | retrives all the contours and reconstructs a full hierarchy of nestted contours
                - contour_mode : string correlating to a key in _modes dictionary
                    - 'none' : cv.CHAIN_APPROX_NONE | stores all contours points without approximating any of them
                    - 'simple' : cv.CHAIN_APPROX_SIMPLE | compress horizontal, vertical, and diagonal segments and leaves only their endpoints. If a contour is straight, only the endpoints are stores
                    - 'tc89_li' : cv.CHAIN_APPROX_TC89_L1 | variant of douglas-peucker algorithm
                    - 'tc9_kcos' : cv.CHAIN_APPROX_TC89_KCOS  | variant of douglas-peucker algorithm
                - filter : boolean value to determine whether or not to filter contours based on threshold values
        '''
        def filter_contours(image_contours:List[MatLike]) -> List[MatLike]:
            '''
                FUNCTIONDOCSTRING
                Arguments:
                    - image_contours : list of all contours in the image
            '''
            filtered_contours = [contour for contour in image_contours if cv.contourArea(contour) > min_threshold and cv.contourArea(contour) < max_threshold]
            return filtered_contours
        _methods:Dict[str, Any] = {
                'external' : cv.RETR_EXTERNAL, # retrieves only external colors, disregarding any contours inside the objects
                'list' : cv.RETR_LIST, # retrives all contours without any hierarchy
                'ccomp' : cv.RETR_CCOMP, # retrieves all contours and organizes them into a two-level hierarchy. Top level contains the outer boundaries of the objects, and the second level contains the boundaries of the inner holes
                'tree' : cv.RETR_TREE # retrives all the contours and reconstructs a full hierarchy of nestted contours
        }
        _modes:Dict[str, Any] = {
                'none' : cv.CHAIN_APPROX_NONE, # stores all contours points without approximating any of them
                'simple' : cv.CHAIN_APPROX_SIMPLE, # compress horizontal, vertical, and diagonal segments and leaves only their endpoints. If a contour is straight, only the endpoints are stores
                'tc89_li' : cv.CHAIN_APPROX_TC89_L1, # variant of douglas-peucker algorithm
                'tc9_kcos' : cv.CHAIN_APPROX_TC89_KCOS  # variant of douglas-peucker algorithm
        }
        _method = _methods.get(contour_method, cv.RETR_TREE)
        _mode = _modes.get(contour_mode, cv.CHAIN_APPROX_SIMPLE)
        contours = cv.findContours(self.ImageData, _method, _mode)[0]
        grayscale = cv.cvtColor(self.OriginalImageData, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(grayscale)
        if filter:
            potential_images = filter_contours(contours)
        else:
            potential_images = contours
        rect_info = []
        for contour in potential_images:
            x, y, w, h = cv.boundingRect(contour)
            r_info = [x, y, w, h]
            rect_info.append(r_info)
            cv.rectangle(self.ImageData, (x, y), (x + w, y + h), 255, -1)
            cv.bitwise_and(self.ImageData, self.ImageData, mask=mask)
        self.MaskRectInfo = rect_info
        self.MaskRectScale = (self.ScaleX, self.ScaleY)
        return rect_info

    @add_transformation('reset and add rectangles to original')
    def reset_and_add_recangles(self) -> MatLike:
        self.reset_image_data(retain_transformations=True)
        rect_scale_x, rect_scale_y = self.MaskRectScale
        self.enlarge_image(rect_scale_x, rect_scale_y)
        buffer = int(self.ImageData.shape[1] * .006)
        for rect in self.MaskRectInfo:
            x, y, w, h = rect
            cv.rectangle(self.ImageData, (x, y), (x + w + buffer*5, y + h + buffer), (255,255,255), -1)
        return self.ImageData


    def draw_text_box_outline(self, confidence_threshold:int=60) -> list:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
        self.reset_mask_rectangles()
        d:dict = pytesseract.image_to_data(self.ImageData, output_type=Output.DICT) # sort image data into dictionary
        n_boxes = len(d['text'])
        drawn_rects:List[Tuple[int]] = []
        for i in range(n_boxes):
            if int(d['conf'][i]) > confidence_threshold:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                if w > (self.ImageData.shape[0] / 2) or h > (self.ImageData.shape[1] / 2):
                    continue
                drawn_rects.append((x,y,w,h))
                self.ImageData = cv.rectangle(self.ImageData, (x,y), (x+w, y+h), 173, 2)
        self.MaskRectInfo = drawn_rects
        self.MaskRectScale = (self.ScaleX, self.ScaleY)
        return [d, drawn_rects]

    def draw_filled_text_box(self, confidence_threshold:int=60) -> list:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
        self.reset_mask_rectangles()
        d:dict = pytesseract.image_to_data(self.ImageData, output_type=Output.DICT) # sort image data into dictionary
        n_boxes = len(d['text'])
        drawn_rects:List[Tuple[int]] = []
        for i in range(n_boxes):
            if int(d['conf'][i]) > confidence_threshold:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                if w > (self.ImageData.shape[0] / 2) or h > (self.ImageData.shape[1] / 2):
                    continue
                drawn_rects.append((x,y,w,h))
                self.ImageData = cv.rectangle(self.ImageData, (x,y), (x+w, y+h), 255, -1)
        self.MaskRectInfo = drawn_rects
        self.MaskRectScale = (self.ScaleX, self.ScaleY)
        return [d, drawn_rects]

    def group_text_by_block(self, d:dict) -> List[str]:
        grouped_text:Dict[str, List[str]] = {}
        for i, text in enumerate(d['text']):
            if text.strip():  # Skip empty text
                block_num = d['block_num'][i]
                block_key = f'{block_num}'
                if block_key not in grouped_text:
                    grouped_text[block_key] = []
                grouped_text[block_key].append(text)
        text_groups = [f'{key}  |  {' '.join(value)}' for key, value in grouped_text.items()]
        return text_groups

    def print_grouped_text(self, text_groups:List[str]) -> None:
        for group in text_groups:
            print(group)

    def formatted_tesseract_dict(self) -> str:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
        formatted_dict:str = pytesseract.image_to_data(self.ImageData, output_type=Output.STRING)
        return formatted_dict
