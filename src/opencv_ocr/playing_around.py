import cv2 as cv
from cv2.typing import MatLike
import numpy as np
from typing import Callable, List, Tuple, Dict, Any
import pytesseract
from pytesseract import Output
from image_to_process import ImageToProcess


def group_text_by_block(d):
    grouped_text:Dict[str, List[str]] = {}
    for i, text in enumerate(d['text']):
        if text.strip():  # Skip empty text
            block_num = d['block_num'][i]
            block_key = f'{block_num}'
            if block_key not in grouped_text:
                grouped_text[block_key] = []
            grouped_text[block_key].append(text)
    for key, value in grouped_text.items():
        print(key, ' '.join(value), sep='  |  ')

def working_for_recipe_card_jpeg():
    test = ImageToProcess('./images/recipe_card.jpeg')
    test.enlarge_image(2, 2)
    test.convert_to_grayscale()
    test.gaussian_blur(blur_kernel_size=131, edge_detect=True)
    test.display_image('gaussian blur')
    test.otsu_threshold(0, 255)
    test.display_image('threshhold and closed and dilated')
    test.canny_edge_threshold(0, 255, 5, True)
    test.display_image('canny')
    test.dilate(iterations=3)
    test.display_image('dilate')
    min_thresh = int((test.OriginalImageData.shape[1]/30)**2)
    max_thresh = int((test.OriginalImageData.shape[1]/2)**2)
    test.draw_contours(min_threshold=min_thresh, max_threshold=max_thresh, contour_method='external', contour_mode='none')
    rect_info = test.draw_contours(min_threshold=min_thresh, max_threshold=max_thresh, contour_method='external', contour_mode='none')
    test.display_image('masked image')
    test.reset_and_add_recangles()
    test.display_image('original grayscale with rects', test.ImageData)
    test.convert_to_grayscale()
    test.bilateral_filter_blur(blur_kernel_size=5)
    test.otsu_threshold(230,255)
    test.close_pixels()
    test.dilate()
    test.bilateral_filter_blur(blur_kernel_size=1)
    test.otsu_threshold(230,255)
    test.display_image('processed with rects')
    # d:dict = pytesseract.image_to_data(test.ImageData, output_type=Output.STRING)
    # print(d)
    d:dict = pytesseract.image_to_data(test.ImageData, output_type=Output.DICT) # sort image data into dictionary
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            test.ImageData = cv.rectangle(test.ImageData, (x,y), (x+w, y+h), (0,255,0), 2)
    test.display_image('extracted text')
    group_text_by_block(d)

def mask_over_v2(path):
    test = ImageToProcess(image_path=path)
    test.enlarge_image(2, 2)
    test.convert_to_grayscale()
    # test.gaussian_blur(blur_kernel_size=131, edge_detect=True)
    # test.display_image('gaussian blur')
    test.otsu_threshold(0, 255)
    # test.display_image('threshhold and closed and dilated')
    test.canny_edge_threshold(0, 255, 5, True)
    test.display_image('canny')
    test.erode(iterations=4)
    test.display_image('eroded')
    min_thresh = int((test.OriginalImageData.shape[1]/35)**2)
    max_thresh = int((test.OriginalImageData.shape[1]*.5)**2)
    test.draw_contours(min_threshold=min_thresh, max_threshold=max_thresh)
    test.display_image('first')
    rect_info = test.draw_contours(min_threshold=min_thresh, max_threshold=max_thresh)
    test.display_image('second image')
    test.reset_image_data()
    test.enlarge_image(2,2)
    print(rect_info)
    buffer = int(test.ImageData.shape[1] * .006)
    for rect in rect_info:
        x, y, w, h = rect
        cv.rectangle(test.ImageData, (x, y), (x + w + buffer*5, y + h + buffer), (255,255,255), -1)
    # test.display_image('original grayscale with rects', test.ImageData)
    test.convert_to_grayscale()
    test.bilateral_filter_blur(blur_kernel_size=5)
    test.otsu_threshold(230,255)
    test.close_pixels()
    test.dilate()
    test.bilateral_filter_blur(blur_kernel_size=1)
    test.otsu_threshold(230,255)
    # test.display_image('processed with rects')
    # d:dict = pytesseract.image_to_data(test.ImageData, output_type=Output.STRING)
    # print(d)
    d:dict = pytesseract.image_to_data(test.ImageData, output_type=Output.DICT) # sort image data into dictionary
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            test.ImageData = cv.rectangle(test.ImageData, (x,y), (x+w, y+h), (0,255,0), 2)
    test.display_image('extracted text')
    group_text_by_block(d)

def mask_over_v3(path):
    test = ImageToProcess(image_path=path)
    test.enlarge_image(2, 2)
    test.convert_to_grayscale()
    # test.gaussian_blur(blur_kernel_size=131, edge_detect=True)
    test.display_image('gaussian blur')
    # test.otsu_threshold(0, 150)
    test.simple_threshold(240, 255)
    test.display_image('threshhold and closed and dilated')
    test.canny_edge_threshold(0, 255, 5, True)
    test.display_image('canny')
    test.erode(iterations=4)
    test.display_image('eroded')
    min_thresh = int((test.OriginalImageData.shape[1]/15)**2)
    max_thresh = int((test.OriginalImageData.shape[1]/2)**2)
    test.draw_contours(min_threshold=min_thresh, max_threshold=max_thresh, contour_method='tree', contour_mode='none')
    rect_info = test.draw_contours(min_threshold=min_thresh, max_threshold=max_thresh, contour_method='tree', contour_mode='none')
    test.display_image('masked image')
    test.reset_image_data()
    test.enlarge_image(2,2)
    print(rect_info)
    buffer = int(test.ImageData.shape[1] * .006)
    for rect in rect_info:
        x, y, w, h = rect
        cv.rectangle(test.ImageData, (x, y), (x + w + buffer*5, y + h + buffer), (255,255,255), -1)
    test.display_image('original grayscale with rects', test.ImageData)
    test.convert_to_grayscale()
    test.bilateral_filter_blur(blur_kernel_size=5)
    test.otsu_threshold(230,255)
    test.close_pixels()
    test.dilate()
    test.bilateral_filter_blur(blur_kernel_size=1)
    test.otsu_threshold(230,255)
    test.display_image('processed with rects')
    # d:dict = pytesseract.image_to_data(test.ImageData, output_type=Output.STRING)
    # print(d)
    d:dict = pytesseract.image_to_data(test.ImageData, output_type=Output.DICT) # sort image data into dictionary
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            test.ImageData = cv.rectangle(test.ImageData, (x,y), (x+w, y+h), (0,255,0), 2)
    test.display_image('extracted text')
    group_text_by_block(d)

def mask_over_v4(path):
    image = ImageToProcess(path)
    image.enlarge_image(2.5, 2.5)
    image.convert_to_grayscale()
    image.dilate(iterations=3)
    image.adaptive_threshold(kernel_size=15)
    image.display_image()
    image.close_pixels()
    image.gaussian_blur(81, edge_detect=True)
    image.canny_edge_threshold(threshold_value1=100, threshold_value2=255, aperture_size=7, l2_gradient=True)
    image.dilate()
    image.display_image()

def mask_over_v5(path):
    image = ImageToProcess(path)
    image.enlarge_image(2, 2)
    image.convert_to_grayscale()
    image.gaussian_blur(blur_kernel_size=15, edge_detect=True)
    image.otsu_threshold(200, 255)
    image.laplacian_filter(kernel_size=5)
    image.close_pixels()
    min_thresh = int((image.OriginalImageData.shape[1]/15)**2)
    max_thresh = int((image.OriginalImageData.shape[1]/2)**2)
    image.draw_contours(min_threshold=min_thresh, max_threshold=max_thresh)
    image.display_image()

if __name__ == "__main__":
    working_for_recipe_card_jpeg()
    # mask_over_v5('./images/blue_apron.png')