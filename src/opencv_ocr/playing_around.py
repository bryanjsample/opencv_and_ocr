import cv2 as cv
from typing import List, Dict
import pytesseract
from pytesseract import Output
from image_to_process import ImageToProcess
from image_to_extract import ImageToExtract




def working_for_recipe_card_jpeg():
    test = ImageToExtract('./images/recipe_card.jpeg')
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
    d = test.draw_text_box_outline()
    test.display_image('extracted text')
    groups = test.group_text_by_block(d)
    test.print_grouped_text(groups)

def mask_over_v6(path):
    image = ImageToExtract(path)
    image.display_image()
    image.enlarge_image()
    image.display_image()
    image.convert_to_grayscale()
    image.display_image()
    image.canny_edge_threshold(threshold_value1=25, threshold_value2=180, aperture_size=5, l2_gradient=False)
    # image.gaussian_blur(blur_kernel_size=31, edge_detect=True)
    image.display_image()
    # image.display_image()
    # image.draw_filled_text_box(confidence_threshold=75)
    # image.adaptive_threshold(255, 3)
    # image.display_image()
    # image.close_pixels()
    # image.display_image()
    # image.median_blur(3, True)
    # image.display_image()

if __name__ == "__main__":
    # working_for_recipe_card_jpeg()
    mask_over_v6('./images/blue_apron.png')