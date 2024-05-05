'''
    module.py
    Author : Bryan Sample

    DESCRIPTION
'''
import cv2 as cv
from image_to_process import ImageToProcess
import pytesseract
from pytesseract import Output
from typing import List, Tuple

class ImageToExtract(ImageToProcess):
    ''''''
    def __init__(self, image_path: str) -> None:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
        super().__init__(image_path)

    def draw_text_box_outline(self, confidence_threshold:int=60) -> list:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
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
        return [d, drawn_rects]

    def draw_filled_text_box(self, confidence_threshold:int=60) -> list:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
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
        return [d, drawn_rects]

    def formatted_tesseract_dict(self) -> str:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
        formatted_dict:str = pytesseract.image_to_data(self.ImageData, output_type=Output.STRING)
        return formatted_dict
