'''
    image_to_process.py
    Author : Bryan Sample

    Class definition of ImageToProcess, which enables processing of an image to improve it's readability for Tesseract OCR
'''

import cv2
import numpy as np
from typing import Callable, List, Tuple, Dict, Any
from textwrap import dedent


def invert_colors_for_processing(processing_func:Callable) -> Callable:
    '''Decorator function to invert pixel values, call processing function, then invert pixel values back to original.'''
    def wrapper_function(image_object:object, *args, **kwargs) -> None:
        image_object.ImageData = cv2.bitwise_not(image_object.ImageData)
        image_object.InvertedColor = True
        processing_func(image_object, *args, **kwargs)
        image_object.ImageData = cv2.bitwise_not(image_object.ImageData)
        image_object.InvertedColor = False
    return wrapper_function

def add_transformation(transformation_function_name:str) -> Callable:
    '''Decorator function to add values into the Transformations property to keep track of which Transformations have been performed on the image.'''
    def perform_transformation(processing_func:Callable) -> Callable:
        def wrapper_function(image_object:object, *args, **kwargs) -> None:
            try:
                processing_func(image_object, *args, **kwargs)
            except Exception as e:
                raise TransformationFailedError(str(e), '\nThe transformation failed....Run the debugger to investigate further.', image_object.ImagePath)
            else:
                print(transformation_function_name, args, kwargs, sep='\n')
                image_object.Transformations = (transformation_function_name, args, kwargs)
        return wrapper_function
    return perform_transformation

def indent(number_of_single_spaces:int=4) -> str:
    return ' ' * number_of_single_spaces

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
        self._image_data:List[cv2.typing.MatLike] = [cv2.imread(image_path), cv2.imread(image_path)]
        self._inverted_color:bool = False
        self._transformations:Dict[int, Tuple[str, List[Tuple[Any]|Dict[str, Any]]]]|None = None

    @property
    def ImagePath(self) -> str:
        return self._image_path

    @property
    def OriginalImageData(self) -> cv2.typing.MatLike:
        '''If image is resized, then original image data will be resized to match.'''
        return self._image_data[0]

    @property
    def ImageData(self) -> cv2.typing.MatLike:
        return self._image_data[1]
    @ImageData.setter
    def ImageData(self, new_data_value:cv2.typing.MatLike) -> None:
        self._image_data[1] = new_data_value

    @property
    def InvertedColor(self) -> bool:
        return self._inverted_color
    @InvertedColor.setter
    def InvertedColor(self, bool_value:bool) -> None:
        self._inverted_color = bool_value

    @property
    def Transformations(self) -> Dict[int, Tuple[str, List[Tuple[Any]|Dict[str, Any]]]]|None:
        return self._transformations
    @Transformations.setter # transformation_function_name:str='', arguments:Tuple[Any]='', keywords:Dict[str, Any]=''
    def Transformations(self, transformation_info:Tuple[str, Tuple[Any], Tuple[str, Any]]) -> None:
        '''
            Adds values into the Transformations property to keep track of which Transformations have been performed on the image.
            Arguments:
                - transformation_info : tuple containing information about the transformation being performed
                    - transformation_function_name :
                    - arguments : 
                    - keywords : 
        '''
        transformation_function_name:str
        arguments:Tuple[Any]
        keywords:Dict[str, Any]
        transformation_function_name, arguments, keywords = transformation_info
        if self._transformations is None:
            self._transformations = {}
            transformation_number = 0
        else:
            transformation_number = list(self._transformations.keys())[-1] + 1
        self._transformations[transformation_number] = ( transformation_function_name , [ arguments, keywords ])
        print(self.Transformations)

    def get_transformations_string(self) -> str:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
        def format_kwargs(kwargs_list:List[dict]) -> List[str]:
            formatted_kwargs:List[str] = []
            for kwarg_dict in kwargs_list:
                current:List[str] = []
                kwargs = list(kwarg_dict.keys())
                values = list(kwarg_dict.values())
                if len(kwargs) < 1 or len(values) < 1:
                    current.append('None')
                else:
                    for i in range(len(kwargs)):
                        kwarg_str = f'{kwargs[i]} = {values[i]}'
                    current.append(kwarg_str)
                formatted_kwargs.append(', '.join([kw for kw in current if kw !='None']))
            return [kwarg_str if kwarg_str != '' else 'None' for kwarg_str in formatted_kwargs]
        
        def get_align_values(number_list:List[int], name_list:List[str], arg_list:List[Any], kwarg_list:List[str]) -> List[int]:
            number_align:int = max([len(str(x)) for x in number_list])
            name_align:int = max([len(str(x)) for x in name_list])
            arg_align:int = max([len(str(x)) for x in arg_list])
            kwarg_align:int = max([len(str(x)) for x in kwarg_list])
            return [number_align, name_align, arg_align, kwarg_align]

        transformation_numbers:List[int] = list(self.Transformations.keys())
        transformation_information:list = [info for info in list(self.Transformations.values())]

        transformation_function_names:List[str] = [info[0] for info in transformation_information]

        transformation_args_and_kwargs:List[Tuple[Any], Dict[str, Any]] = [info[1] for info in transformation_information]
        transformation_args:List[List[Any]] = [args_and_kwargs[0] if len(args_and_kwargs[0]) != 0 else 'None' for args_and_kwargs in transformation_args_and_kwargs]
        transformation_kwargs:List[dict] = [args_and_kwargs[1] for args_and_kwargs in transformation_args_and_kwargs]
        formatted_kwargs:List[str] = format_kwargs(transformation_kwargs)

        number_align, name_align, arg_align, kwarg_align = get_align_values(transformation_numbers, transformation_function_names, transformation_args, formatted_kwargs)

        transformation_lines = [f'{transformation_numbers[i]: <{number_align}} | {transformation_function_names[i]: <{name_align}} | arguments : {transformation_args[i]: <{arg_align}} | keywords : {formatted_kwargs[i]: <{kwarg_align}}' for i in range(len(transformation_numbers))]
        return f'Transformations Performed:\n{'\n'.join(transformation_lines)}'


    def __str__(self) -> str:
        string_representation = f'''\
Image Path : {self.ImagePath}
    {self.ImageData.shape=}
    {self.ImageData.dtype=}
    {self.OriginalImageData.shape=}
    {self.OriginalImageData.dtype=}

{self.get_transformations_string()}'''
        return string_representation

    def __repr__(self) -> str:
        string_representation = f'''\
Image Path : {self.ImagePath}
    {self.ImageData.shape=}
    {self.ImageData.dtype=}
    {self.OriginalImageData.shape=}
    {self.OriginalImageData.dtype=}

{self.get_transformations_string()}'''
        return string_representation

    def reset_image_data(self):
        self._image_data[0] = cv2.imread(self.ImagePath)
        self.ImageData = cv2.imread(self.ImagePath)
        self.InvertedColor = False
        self.Transformations = None

    def display_image(self, title:str|None=None, image:cv2.typing.MatLike|None=None) -> None:
        if title is None:
            title = self.ImagePath
        if image is None:
            image:cv2.typing.MatLike = self.ImageData
        cv2.imshow(title, image)
        cv2.waitKey(0)

    @add_transformation('shrink image')
    def shrink_image(self, scale_x:float=.75, scale_y:float=.75) -> None:
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
            self._image_data[0] = cv2.resize(self.OriginalImageData, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)

    @add_transformation('enlarge image')
    def enlarge_image(self, scale_x:float=1.5, scale_y:float=1.5) -> None:
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
            self._image_data[0] = cv2.resize(self.OriginalImageData, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)

    @add_transformation('convert to grayscale')
    def convert_to_grayscale(self) -> None:
        self.ImageData = cv2.cvtColor(self.ImageData, cv2.COLOR_BGR2GRAY)

    @add_transformation('standard blur')
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

    @add_transformation('gaussian blur')
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

    @add_transformation('median blur')
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

    @add_transformation('bilateral filter blur')
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

    @add_transformation('simple threshold')
    def simple_threshold(self, threshold_value:int=127, color_if_less_than_threshold:int=255) -> None:
        '''
            If the pixel value is greater than the threshold, it becomes black. If less, it becomes color_if_less_than_threshold.
            Arguments:
                - threshold_value : if average value of pixel exceeds this, it becomes black.
                - color_if_less_than_threshold : grayscale value to set pixel if less than threshold_value
        '''
        if threshold_value >= 250:
            raise InvalidThresholdArgumentsError('Threshold value cannot be greater than or equal to 250!', self.ImagePath)
        else:
            self.ImageData = cv2.threshold(self.ImageData, threshold_value, color_if_less_than_threshold, cv2.THRESH_BINARY)[1]

    @add_transformation('adaptive threshold')
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

    @add_transformation('otsu threshold')
    def otsu_threshold(self) -> None:
        '''
            ! WORKS WELL WITH BIMODAL IMAGES, BUT MAY FAIL TO BINARIZE IMAGES THAT ARE NOT BIMODAL !
            Picks a threshold value that is within the peaks of a images histogram.
        '''
        self.ImageData = cv2.threshold(self.ImageData, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    @add_transformation('dilate')
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

    @add_transformation('erode')
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

    @add_transformation('open pixels (erode then dilate)')
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

    @add_transformation('close pixels (dilate then erode)')
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

    @add_transformation('draw contours')
    def draw_contours(self, contour_area_threshold:int=1000) -> None:

        self.enlarge_image_size(scale_x=2, scale_y=2)
        self.convert_to_grayscale()
        self.median_blur()
        self.simple_threshold(threshold_value=170, color_if_less_than_threshold=255)
        self.open_pixels()
        self.otsu_threshold()

        self.display_image()

        potential_images:list[cv2.typing.MatLike] = []
        contours = cv2.findContours(self.ImageData, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > contour_area_threshold]
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <=1.1:
                potential_images.append(contour)
        self.reset_image_data()
        self.convert_to_grayscale()
        mask = np.zeros_like(self.ImageData)
        for contour in potential_images:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(self.ImageData, (x, y), (x + w, y + h), 255, -1)

        masked_image = cv2.bitwise_and(self.OriginalImageData, self.ImageData, mask=mask)

        self.display_image('masked', masked_image)
        # self.ImageData = cv2.drawContours(self.ImageData, contours, -1, (0, 255, 0), 3)

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
    '''Raised when the threshold is greater than or equal to 250 or the size of neighborhood is greater than or equal to 45 in an adaptive threshold.'''
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

class TransformationFailedError(Exception):
    def __init__(self, original_error_message:str, error_message:str, image_to_process_path:ImageToProcess.ImagePath) -> None:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                - original_error_message : error message from the origin error (except Error as e)
                - custom_error_message : string that contains \\n for each new line followed by desired whitespace. Ex : "Blur kernel size cannot be greater than or equal to 10!\\nAnything greater than 10 will result in an image quality that is too poor.\\nTypically a smaller value will produce better results."
                - image_to_process_path : ImagePath property of ImageToProcess
        '''
        formatted_custom_error_message = format_exception_new_lines(self, error_message, image_to_process_path)
        super().__init__(original_error_message + formatted_custom_error_message)

def main():
    test = ImageToProcess(image_path='recipe_card.jpeg')
    print(test)


if __name__ == "__main__":
    main()