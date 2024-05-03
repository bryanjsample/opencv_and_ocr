'''
    image_to_process.py
    Author : Bryan Sample

    Class definition of ImageToProcess, which enables processing of an image to improve it's readability for Tesseract OCR
'''

import cv2
import numpy as np
from typing import Callable


def invert_colors_for_processing(processing_func:Callable) -> Callable:
    '''Decorator function to invert pixel values, call processing function, then invert pixel values back to original.'''
    def wrapper_function(self, *args, **kwargs) -> None:
        self.ImageData = cv2.bitwise_not(self.ImageData)
        self.InvertedColor = True
        processing_func(self, *args, **kwargs)
        self.ImageData = cv2.bitwise_not(self.ImageData)
        self.InvertedColor = False
    return wrapper_function

class ImageToProcess():
    '''Object representing an image file to be processed to improve readability for Tesseract OCR.'''
    # convert_grayscale:bool=True, remove_noise:bool=True, threshold:bool=False, dilate:bool=False, erode:bool=False, canny_edge:bool=False
    def __init__(self, image_path:str) -> None:
        '''
            Processes image to improve readability for Tesseract OCR.
            Arguments:
                - image_path : path to image being processed
        '''
        self._image_path:str = image_path
        self._image_data:cv2.typing.MatLike = cv2.imread(image_path)
        self._inverted_color:bool = False

    @property
    def ImagePath(self) -> str:
        return self._image_path

    @property
    def ImageData(self) -> cv2.typing.MatLike:
        return self._image_data
    @ImageData.setter
    def ImageData(self, new_data_value):
        self._image_data = new_data_value

    @property
    def InvertedColor(self) -> bool:
        return self._inverted_color
    @InvertedColor.setter
    def InvertedColor(self, bool_value:bool) -> None:
        self._inverted_color = bool_value

    def __str__(self) -> str:
        return f'\n    Image Path : {self.ImagePath}\n'

    def __repr__(self) -> str:
        return f'\n    Image Path : {self.ImagePath}\n'

    def shrink_image_size(self, scale_x:float=.75, scale_y:float=.75) -> None:
        '''
            Shrinks an image to a scale between (0, 1) on the x and y axis. Recommended to input the same scale for x and y to maintain image integrity.
            Arguments:
                - scale_x : scaling of the x axis, must not be greater than or equal to 1.0
                - scale_y : scaling of the y axis, must not be greater than or equal to 1.0
        '''
        if scale_x >= 1 or scale_y >= 1:
            raise InvalidScaleArgumentsError("Scale cannot be greater than or equal to 1.0!", self.ImagePath)
        else:
            self.ImageData = cv2.resize(self.ImageData, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)

    def enlarge_image_size(self, scale_x:float=1.5, scale_y:float=1.5) -> None:
        '''
            Enlarges an image to a scale between (0, 1) on the x and y axis. Recommended to input the same scale for x and y to maintain image integrity.
            Arguments:
                - scale_x : scaling of the x axis, must not be less than or equal to 1.0
                - scale_y : scaling of the y axis, must not be less than or equal to 1.0
        '''
        if scale_x <= 1 or scale_y <= 1:
            raise InvalidScaleArgumentsError("Scale cannot be less than or equal to 1.0!", self.ImagePath)
        else:
            self.ImageData = cv2.resize(self.ImageData, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)

    def convert_to_grayscale(self) -> None:
        self.ImageData = cv2.cvtColor(self.ImageData, cv2.COLOR_BGR2GRAY)

    def standard_blur(self, blur_kernel_size:int=3) -> None:
        '''
            Replaces a pixels value with the average of all pixels under the area (ksize, ksize).
            Arguments:
                - blur_kernel_size : size of the kernel to blur, must not be greater than 9
        '''
        if blur_kernel_size % 2 != 1:
            raise InvalidBlurArgumentsError('Blur kernel size must be odd!', self.ImagePath, kernel_size_error=True)
        elif blur_kernel_size > 9:
            raise InvalidBlurArgumentsError('Blur kernel size cannot be greater than 9!', self.ImagePath)
        else:
            self.ImageData = cv2.blur(self.ImageData, (blur_kernel_size, blur_kernel_size))

    def gaussian_blur(self, blur_kernel_size:int=3, sigma_space:int=0) -> None:
        '''
            Reduces gaussian noise and reduces image detail. DOES NOT PRESERVE EDGES!
            Arguments:
                - blur_kernel_size : size of the kernel to blur, must not be greater than 9
                - sigma_space : standard deviation of blur on the x axis, must not be greater than 3
        '''
        if blur_kernel_size % 2 != 1:
            raise InvalidBlurArgumentsError('Blur kernel size must be odd!', self.ImagePath, kernel_size_error=True)
        elif blur_kernel_size > 9:
            raise InvalidBlurArgumentsError('Blur kernel size cannot be greater than 9!', self.ImagePath)
        elif sigma_space > 3:
            raise InvalidBlurArgumentsError('Sigma value of a gaussian blur cannot be greater than 3!', self.ImagePath)
        else:
            self.ImageData = cv2.GaussianBlur(self.ImageData, (blur_kernel_size, blur_kernel_size), sigma_space)

    def median_blur(self, blur_kernel_size:int=3) -> None:
        '''
            ! BEST OPTION FOR SPEED AND EDGE PRESERVATION !
            Replaces the pixel values with the median value available in the neighborhood values.
            Preserves edges as the median value must be the value of one of neighboring pixels, thus is the best option for removing salt-and-pepper noise.
            Arguments:
                - blur_kernel_size : size of the kernel to blur, must not be greater than 9
        '''
        if blur_kernel_size % 2 != 1:
            raise InvalidBlurArgumentsError('Blur kernel size must be odd!', self.ImagePath, kernel_size_error=True)
        elif blur_kernel_size > 9:
            raise InvalidBlurArgumentsError('Blur kernel size cannot be greater than 9!', self.ImagePath)
        else:
            self.ImageData = cv2.medianBlur(self.ImageData, blur_kernel_size)

    def bilateral_filter_blur(self, blur_kernel_size:int=15, sigma_color:int=75, sigma_space:int=75) -> None:
        '''
            ! GOOD OPTION FOR EDGE PRESERVATION, BUT LACKS IN SPEED !
            Applies a normalization factor to a gaussian blur, ensuring that only pixels with similar intensity to the central pixel are blurred.
            Preserves edges as the edges that have larger intensity variation are preserved.
            Arguments:
                - blur_kernel_size : size of the kernel to blur, must not be greater than 25
                - sigma_color : normalization factor of colors
                - simga_space : normalization factor of space
        '''
        if blur_kernel_size % 2 != 1:
            raise InvalidBlurArgumentsError('Blur kernel size must be odd!', self.ImagePath, kernel_size_error=True)
        elif blur_kernel_size > 25:
            raise InvalidBlurArgumentsError('Blur kernel size cannot be greater than or equal to 25!', self.ImagePath)
        else:
            self.ImageData = cv2.bilateralFilter(self.ImageData, blur_kernel_size, sigma_color, sigma_space)

    def simple_threshold(self, threshold_value:int=127, color_if_less_than_threshold:int=255) -> None:
        '''
            If the pixel value is greater than the threshold, it becomes black. If less, it becomes color_if_less_than_threshold.
            Arguments:
                - threshold_value : if average value of pixel exceeds this, it becomes black.
                - color_if_less_than_threshold : grayscale value to set pixel if less than threshold_value
        '''
        if threshold_value >= 175:
            raise InvalidThresholdArgumentsError('Threshold value cannot be greater than or equal to 175!', self.ImagePath)
        else:
            self.ImageData = cv2.threshold(self.ImageData, threshold_value, color_if_less_than_threshold, cv2.THRESH_BINARY)[1]

    def adaptive_threshold(self, color_if_less_than_threshold:int=255, kernel_size:int=31, constant:int=2) -> None:
        '''
            Allows an algorithm to calculate the threshold for small regions of the image.
            Arguments:
                - color_if_less_than_threshold : grayscale value to set pixel if less than threshold_value
                - kernel_size : area of pixels to calculate algorithm within
                - constant : constant to be subtracted from each result
        '''
        if kernel_size % 2 != 1:
            raise InvalidThresholdArgumentsError('Blur kernel size must be odd!', self.ImagePath)
        if kernel_size >= 45:
            raise InvalidThresholdArgumentsError('Size of neighborhood cannot be greater than or equal to 45!', self.ImagePath)
        else:
            self.ImageData = cv2.adaptiveThreshold(self.ImageData, color_if_less_than_threshold, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kernel_size, constant)

    def otsu_threshold(self) -> None:
        '''
            ! WORKS WELL WITH BIMODAL IMAGES, BUT MAY FAIL TO BINARIZE IMAGES THAT ARE NOT BIMODAL !
            Picks a threshold value that is within the peaks of a images histogram.
        '''
        self.ImageData = cv2.threshold(self.ImageData, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    @invert_colors_for_processing
    def dilate(self, kernel_size:int=5, iterations:int=1) -> None:
        '''
            ! USEFUL TO DILATE AFTER ERODING !
            Increases the boundaries of a foreground object and accentuates features of the image.
            Arguments:
                - kernel_size : area of pixels to calculate algorithm within
                - iterations : number of iterations to run
        '''
        if not self.InvertedColor:
            raise InvalidDilateOrErodeArguments('Image must be inverted for dilate to work well!', self.ImagePath)
        elif kernel_size % 2 != 1:
            raise InvalidDilateOrErodeArguments('Blur kernel size must be odd!', self.ImagePath)
        elif kernel_size > 9:
            raise InvalidDilateOrErodeArguments('Size of neighborhood cannot be greater than 9!', self.ImagePath)
        elif iterations >= 5:
            raise InvalidDilateOrErodeArguments('Number of iterations cannot be greater than or equal to 5!', self.ImagePath)
        else:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            self.ImageData = cv2.dilate(self.ImageData, kernel, iterations=iterations)

    @invert_colors_for_processing
    def erode(self, kernel_size:int=5, iterations:int=1) -> None:
        '''
            ! USEFUL TO DILATE AFTER ERODING !
            Decreases the boundaries of a foreground object and diminishes features of the image.
            Arguments:
                - kernel_size : area of pixels to calculate algorithm within
                - iterations : number of iterations to run
        '''
        if not self.InvertedColor:
            raise InvalidDilateOrErodeArguments('Image must be inverted for erode to work well!', self.ImagePath)
        elif kernel_size % 2 != 1:
            raise InvalidDilateOrErodeArguments('Blur kernel size must be odd!', self.ImagePath)
        elif kernel_size > 9:
            raise InvalidDilateOrErodeArguments('Size of neighborhood cannot be greater than 9!', self.ImagePath)
        elif iterations >= 5:
            raise InvalidDilateOrErodeArguments('Number of iterations cannot be greater than or equal to 5!', self.ImagePath)
        else:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            self.ImageData = cv2.erode(self.ImageData, kernel, iterations=iterations)

    @invert_colors_for_processing
    def open_pixels(self, kernel_size:int=3) -> None:
        '''
            Erosion followed by dilation. Useful in removing noise.
            Arguments:
                - kernel_size : area of pixels to calculate algorithm within
        '''
        if not self.InvertedColor:
            raise InvalidDilateOrErodeArguments('Image must be inverted for opening to work well!', self.ImagePath)
        elif kernel_size % 2 != 1:
            raise InvalidDilateOrErodeArguments('Blur kernel size must be odd!', self.ImagePath)
        elif kernel_size > 9:
            raise InvalidDilateOrErodeArguments('Size of neighborhood cannot be greater than 9!', self.ImagePath)
        else:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            self.ImageData = cv2.morphologyEx(self.ImageData, cv2.MORPH_OPEN, kernel)

    @invert_colors_for_processing
    def close_pixels(self, kernel_size:int=3) -> None:
        '''
            Dilation followed by erosion. Useful in closing small holse inside foreground objects.
            Arguments:
                - kernel_size : area of pixels to calculate algorithm within
        '''
        if not self.InvertedColor:
            raise InvalidDilateOrErodeArguments('Image must be inverted for closing to work well!', self.ImagePath)
        elif kernel_size % 2 != 1:
            raise InvalidDilateOrErodeArguments('Blur kernel size must be odd!', self.ImagePath)
        elif kernel_size > 9:
            raise InvalidDilateOrErodeArguments('Size of neighborhood cannot be greater than 9!', self.ImagePath)
        else:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            self.ImageData = cv2.morphologyEx(self.ImageData, cv2.MORPH_CLOSE, kernel)

# error handling definitions

def format_exception_new_lines(exception_class:object, error_message:str, image_to_process_path:ImageToProcess.ImagePath) -> str:
    '''
        Formats an error message nicely to align left along the same vertical column for the entirety of the error message.
        Arguments:
            - exception_class : class object for the error, pass in self (used to determine left spacing)
            - error_message : string that contains \\n for each new line followed by desired whitespace. Ex : "Blur kernel size cannot be greater than or equal to 10!\\nAnything greater than 10 will result in an image quality that is too poor.\\nTypically a smaller value will produce better results."
    '''
    new_line =  ' ' * len(f'{exception_class.__class__.__name__}: ')
    error_message_lines = error_message.split('\n')
    error_message_lines.extend(['', f'Image that caused failure : {image_to_process_path}'])
    return f'\n{new_line}'.join(error_message_lines)

class InvalidScaleArgumentsError(Exception):
    '''Raised when the scale is greater than or equal to 1.0 when shrinking an image or less than or equal to 1.0 when enlarging an image.'''
    def __init__(self, error_message:str, image_to_process_path:ImageToProcess.ImagePath) -> None:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                - error_message : string that contains \\n for each new line followed by desired whitespace. Ex : "Blur kernel size cannot be greater than or equal to 10!\\nAnything greater than 10 will result in an image quality that is too poor.\\nTypically a smaller value will produce better results."
                - image_to_process_path : ImagePath property of ImageToProcess
        '''
        formatted_error_message = format_exception_new_lines(self, error_message, image_to_process_path)
        super().__init__(formatted_error_message)

class InvalidBlurArgumentsError(Exception):
    '''Raised when the blur kernel size is greater than 9 (or 25 in a bilateral filter) or the sigma value of a gaussian blur is greater than 3'''
    def __init__(self, error_message:str, image_to_process_path:ImageToProcess.ImagePath, kernel_size_error:bool=False) -> None:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                - error_message : string that contains \\n for each new line followed by desired whitespace. Ex : "Blur kernel size cannot be greater than or equal to 10!\\nAnything greater than 10 will result in an image quality that is too poor.\\nTypically a smaller value will produce better results."
                - image_to_process_path : ImagePath property of ImageToProcess
        '''
        if not kernel_size_error:
            error_message += '\nAnything greater will result in an image quality that is too poor.\nTypically a smaller value will produce better results.'
        formatted_error_message = format_exception_new_lines(self, error_message, image_to_process_path)
        super().__init__(formatted_error_message)

class InvalidThresholdArgumentsError(Exception):
    '''Raised when the threshold is greater than or equal to 175 or the size of neighborhood is greater than or equal to 45 in an adaptive threshold.'''
    def __init__(self, error_message:str, image_to_process_path:ImageToProcess.ImagePath) -> None:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                - error_message : string that contains \\n for each new line followed by desired whitespace. Ex : "Blur kernel size cannot be greater than or equal to 10!\\nAnything greater than 10 will result in an image quality that is too poor.\\nTypically a smaller value will produce better results."
                - image_to_process_path : ImagePath property of ImageToProcess
        '''
        formatted_error_message = format_exception_new_lines(self, error_message, image_to_process_path)
        super().__init__(formatted_error_message)

class InvalidDilateOrErodeArguments(Exception):
    '''Raised when the neighborhood size is greater than or equal to 10 when eroding or dilating.'''
    def __init__(self, error_message:str, image_to_process_path:ImageToProcess.ImagePath) -> None:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                - error_message : string that contains \\n for each new line followed by desired whitespace. Ex : "Blur kernel size cannot be greater than or equal to 10!\\nAnything greater than 10 will result in an image quality that is too poor.\\nTypically a smaller value will produce better results."
                - image_to_process_path : ImagePath property of ImageToProcess
        '''
        formatted_error_message = format_exception_new_lines(self, error_message, image_to_process_path)
        super().__init__(formatted_error_message)

def main():
    test = ImageToProcess(image_path='recipe_card.jpeg')
    test.enlarge_image_size(scale_x=2, scale_y=2)
    test.convert_to_grayscale()
    test.median_blur()
    test.otsu_threshold()
    test.adaptive_threshold()
    test.erode()
    test.dilate()
    cv2.imshow('test', test.ImageData)
    cv2.waitKey(0)
    cv2.imwrite('dilate.png', test.ImageData)


if __name__ == "__main__":
    main()