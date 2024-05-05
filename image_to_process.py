'''
    image_to_process.py
    Author : Bryan Sample

    Class definition of ImageToProcess, which enables processing of an image to improve it's readability for Tesseract OCR
'''

import cv2 as cv
from cv2.typing import MatLike
import numpy as np
from typing import Callable, List, Tuple, Dict, Any
import pytesseract
from pytesseract import Output

# decorator function definitions

def invert_colors_for_processing(processing_func:Callable) -> Callable:
    '''Decorator function to invert pixel values, call processing function, then invert pixel values back to original.'''
    def wrapper_function(image_object:object, *args, **kwargs) -> None:
        image_object.ImageData = cv.bitwise_not(image_object.ImageData)
        image_object.InvertedColor = True
        processing_func(image_object, *args, **kwargs)
        image_object.ImageData = cv.bitwise_not(image_object.ImageData)
        image_object.InvertedColor = False
    return wrapper_function

def add_transformation(transformation_function_name:str) -> Callable:
    '''Decorator function to add values into the Transformations property to keep track of which Transformations have been performed on the image.'''
    def perform_transformation(processing_func:Callable) -> Callable:
        def wrapper_function(image_object:object, *args, **kwargs) -> None:
            try:
                return_val = processing_func(image_object, *args, **kwargs)
            except Exception as e:
                print(image_object)
                raise TransformationFailedError(str(e), '\nThe transformation failed....Run the debugger to investigate further.', image_object.ImagePath)
            else:
                image_object.Transformations = (transformation_function_name, args, kwargs)
            return return_val
        return wrapper_function
    return perform_transformation

def indent(number_of_single_spaces:int=4) -> str:
    return ' ' * number_of_single_spaces

# main class definition
class ImageToProcess():
    '''Object representing an image file to be processed to improve readability for Tesseract OCR.'''
    def __init__(self, image_path:str) -> None:
        '''
            Processes image to improve readability for Tesseract OCR.
            Arguments:
                - image_path : path to image being processed
        '''
        self._image_path:str = image_path
        self._image_data:List[MatLike] = [cv.imread(image_path), cv.imread(image_path)]
        self._inverted_color:bool = False
        self._transformations:Dict[int, Tuple[str, List[Tuple[Any]|Dict[str, Any]]]]|None = None

    @property
    def ImagePath(self) -> str:
        return self._image_path

    @property
    def OriginalImageData(self) -> MatLike:
        '''If image is resized, then original image data will be resized to match.'''
        return self._image_data[0]

    @property
    def ImageData(self) -> MatLike:
        return self._image_data[1]
    @ImageData.setter
    def ImageData(self, new_data_value:MatLike) -> None:
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
    @Transformations.setter
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

    def get_transformations_string(self) -> str:
        '''Format information stored in self.Transformations to be printed'''
        def format_kwargs(kwargs_list:List[dict]) -> List[str]:
            '''
                FUNCTIONDOCSTRING
                Arguments:
                    -
            '''
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
        def format_args(args_list:List[Tuple|str]) -> List[str]:
            formatted_args:List[str] = []
            for args in args_list:
                if args == 'None':
                    formatted_args.append(args)
                else:
                    formatted_args.append(', '.join([str(arg) for arg in args]))
            return formatted_args
        def get_align_values(number_list:List[int], name_list:List[str], arg_list:List[Any], kwarg_list:List[str]) -> List[int]:
            '''
                FUNCTIONDOCSTRING
                Arguments:
                    -
            '''
            number_align:int = max([len(str(x)) for x in number_list])
            name_align:int = max([len(str(x)) for x in name_list])
            arg_align:int = max([len(str(x)) for x in arg_list])
            kwarg_align:int = max([len(str(x)) for x in kwarg_list])
            return [number_align, name_align, arg_align, kwarg_align]
        if self.Transformations is None:
            return f'Transformations Performed:\n    None'
        # obtain key and value from self.Transformations
        transformation_numbers:List[int] = list(self.Transformations.keys())
        transformation_information:list = [info for info in list(self.Transformations.values())]
        # obtain function names from self.Transformations.Values.Keys
        transformation_function_names:List[str] = [info[0] for info in transformation_information]
        # obtain args and kwargs from self.Transformations.Values.Values
        transformation_args_and_kwargs:List[Tuple[Any], Dict[str, Any]] = [info[1] for info in transformation_information]
        transformation_args:List[List[Any]] = [args_and_kwargs[0] if len(args_and_kwargs[0]) != 0 else 'None' for args_and_kwargs in transformation_args_and_kwargs]
        transformation_kwargs:List[dict] = [args_and_kwargs[1] for args_and_kwargs in transformation_args_and_kwargs]
        # format args and kwargs for output
        formatted_args = format_args(transformation_args)
        formatted_kwargs:List[str] = format_kwargs(transformation_kwargs)
        # obtain max item length from each list to align columns
        number_align, name_align, arg_align, kwarg_align = get_align_values(transformation_numbers, transformation_function_names, transformation_args, formatted_kwargs)
        # form a list of lines, one for each transformation
        transformation_lines = [f'{transformation_numbers[i]: <{number_align}} | {transformation_function_names[i]: <{name_align}} | arguments : {formatted_args[i]: <{arg_align}} | keywords : {formatted_kwargs[i]: <{kwarg_align}}' for i in range(len(transformation_numbers))]
        return f'Transformations Performed:\n{'\n'.join(transformation_lines)}' # return joined lines

    def __str__(self) -> str:
        '''
Image Path : recipe_card.jpeg
    self.ImageData.shape=(2263, 1780)
    self.ImageData.dtype=dtype('uint8')
    self.OriginalImageData.shape=(2263, 1780, 3)
    self.OriginalImageData.dtype=dtype('uint8')

Transformations Performed:
0 | shrink image                     | arguments : None | keywords : None                 
1 | convert to grayscale             | arguments : None | keywords : None                 
2 | bilateral filter blur            | arguments : None | keywords : None                 
3 | otsu threshold                   | arguments : None | keywords : None                 
4 | simple threshold                 | arguments : None | keywords : threshold_value = 175
5 | close pixels (dilate then erode) | arguments : None | keywords : None 
        '''
        string_representation = f'''\
Image Path : {self.ImagePath}
    {self.ImageData.shape=}
    {self.ImageData.dtype=}
    {self.OriginalImageData.shape=}
    {self.OriginalImageData.dtype=}

{self.get_transformations_string()}
'''
        return string_representation

    def __repr__(self) -> str:
        '''
Image Path : recipe_card.jpeg
    self.ImageData.shape=(2263, 1780)
    self.ImageData.dtype=dtype('uint8')
    self.OriginalImageData.shape=(2263, 1780, 3)
    self.OriginalImageData.dtype=dtype('uint8')

Transformations Performed:
0 | shrink image                     | arguments : None | keywords : None                 
1 | convert to grayscale             | arguments : None | keywords : None                 
2 | bilateral filter blur            | arguments : None | keywords : None                 
3 | otsu threshold                   | arguments : None | keywords : None                 
4 | simple threshold                 | arguments : None | keywords : threshold_value = 175
5 | close pixels (dilate then erode) | arguments : None | keywords : None 
        '''
        string_representation = f'''\
Image Path : {self.ImagePath}
    {self.ImageData.shape=}
    {self.ImageData.dtype=}
    {self.OriginalImageData.shape=}
    {self.OriginalImageData.dtype=}

{self.get_transformations_string()}
'''
        return string_representation

    def reset_image_data(self) -> None:
        self._image_data[0] = cv.imread(self.ImagePath)
        self.ImageData = cv.imread(self.ImagePath)
        self.InvertedColor = False
        self._transformations = None

    def display_image(self, title:str|None=None, image:MatLike|None=None) -> MatLike:
        if title is None:
            title = self.ImagePath
        if image is None:
            image:MatLike = self.ImageData
        cv.imshow(title, image)
        print(title, '\n\n' f'{self}')
        cv.waitKey(0)
        cv.destroyAllWindows()
        return image

    @add_transformation('shrink image')
    def shrink_image(self, scale_x:float=.75, scale_y:float=.75 ) -> MatLike:
        '''
            Shrinks an image to a scale between (0, 1) on the x and y axis. Recommended to input the same scale for x and y to maintain image integrity.
            Arguments:
                - scale_x : scaling of the x axis, must not be greater than or equal to 1.0
                - scale_y : scaling of the y axis, must not be greater than or equal to 1.0
        '''
        if scale_x >= 1 or scale_y >= 1:
            raise InvalidScaleArgumentsError("Scale cannot be greater than or equal to 1.0!", self.ImagePath)
        else:
            self.ImageData = cv.resize(self.ImageData, None, fx=scale_x, fy=scale_y, interpolation=cv.INTER_AREA)
            self._image_data[0] = cv.resize(self.OriginalImageData, None, fx=scale_x, fy=scale_y, interpolation=cv.INTER_AREA)
            return self.ImageData

    @add_transformation('enlarge image')
    def enlarge_image(self, scale_x:float=1.5, scale_y:float=1.5) -> MatLike:
        '''
            Enlarges an image to a scale between (0, 1) on the x and y axis. Recommended to input the same scale for x and y to maintain image integrity.
            Arguments:
                - scale_x : scaling of the x axis, must not be less than or equal to 1.0
                - scale_y : scaling of the y axis, must not be less than or equal to 1.0
        '''
        if scale_x <= 1 or scale_y <= 1:
            raise InvalidScaleArgumentsError("Scale cannot be less than or equal to 1.0!", self.ImagePath)
        else:
            self.ImageData = cv.resize(self.ImageData, None, fx=scale_x, fy=scale_y, interpolation=cv.INTER_CUBIC)
            self._image_data[0] = cv.resize(self.OriginalImageData, None, fx=scale_x, fy=scale_y, interpolation=cv.INTER_CUBIC)
            return self.ImageData

    @add_transformation('convert to grayscale')
    def convert_to_grayscale(self) -> MatLike:
        self.ImageData = cv.cvtColor(self.ImageData, cv.COLOR_BGR2GRAY)
        return self.ImageData

    @add_transformation('standard blur')
    def standard_blur(self, blur_kernel_size:int=3, edge_detect:bool=False) -> MatLike:
        '''
            Replaces a pixels value with the average of all pixels under the area (ksize, ksize).
            Arguments:
                - blur_kernel_size : size of the kernel to blur, must not be greater than 9
        '''
        if edge_detect:
            pass
        elif blur_kernel_size % 2 != 1:
            raise InvalidBlurArgumentsError('Blur kernel size must be odd!', self.ImagePath, kernel_size_error=True)
        elif blur_kernel_size > 9:
            raise InvalidBlurArgumentsError('Blur kernel size cannot be greater than 9!', self.ImagePath)
        self.ImageData = cv.blur(self.ImageData, (blur_kernel_size, blur_kernel_size))
        return self.ImageData

    @add_transformation('gaussian blur')
    def gaussian_blur(self, blur_kernel_size:int=3, sigma_space:int=0, edge_detect:bool=False) -> MatLike:
        '''
            Reduces gaussian noise and reduces image detail. DOES NOT PRESERVE EDGES!
            Arguments:
                - blur_kernel_size : size of the kernel to blur, must not be greater than 9
                - sigma_space : standard deviation of blur on the x axis, must not be greater than 3
        '''
        if edge_detect:
            pass
        elif blur_kernel_size % 2 != 1:
            raise InvalidBlurArgumentsError('Blur kernel size must be odd!', self.ImagePath, kernel_size_error=True)
        elif blur_kernel_size > 9:
            raise InvalidBlurArgumentsError('Blur kernel size cannot be greater than 9!', self.ImagePath)
        if sigma_space > 3:
            raise InvalidBlurArgumentsError('Sigma value of a gaussian blur cannot be greater than 3!', self.ImagePath)
        self.ImageData = cv.GaussianBlur(self.ImageData, (blur_kernel_size, blur_kernel_size), sigma_space)
        return self.ImageData

    @add_transformation('median blur')
    def median_blur(self, blur_kernel_size:int=3, edge_detect:bool=False) -> MatLike:
        '''
            ! BEST OPTION FOR SPEED AND EDGE PRESERVATION !
            Replaces the pixel values with the median value available in the neighborhood values.
            Preserves edges as the median value must be the value of one of neighboring pixels, thus is the best option for removing salt-and-pepper noise.
            Arguments:
                - blur_kernel_size : size of the kernel to blur, must not be greater than 9
        '''
        if edge_detect:
            pass
        elif blur_kernel_size % 2 != 1:
            raise InvalidBlurArgumentsError('Blur kernel size must be odd!', self.ImagePath, kernel_size_error=True)
        elif blur_kernel_size > 9:
            raise InvalidBlurArgumentsError('Blur kernel size cannot be greater than 9!', self.ImagePath)
        self.ImageData = cv.medianBlur(self.ImageData, blur_kernel_size)
        return self.ImageData

    @add_transformation('bilateral filter blur')
    def bilateral_filter_blur(self, blur_kernel_size:int=15, sigma_color:int=75, sigma_space:int=75, edge_detect:bool=False) -> MatLike:
        '''
            ! GOOD OPTION FOR EDGE PRESERVATION, BUT LACKS IN SPEED !
            Applies a normalization factor to a gaussian blur, ensuring that only pixels with similar intensity to the central pixel are blurred.
            Preserves edges as the edges that have larger intensity variation are preserved.
            Arguments:
                - blur_kernel_size : size of the kernel to blur, must not be greater than 25
                - sigma_color : normalization factor of colors
                - simga_space : normalization factor of space
        '''
        if edge_detect:
            pass
        elif blur_kernel_size % 2 != 1:
            raise InvalidBlurArgumentsError('Blur kernel size must be odd!', self.ImagePath, kernel_size_error=True)
        elif blur_kernel_size > 25:
            raise InvalidBlurArgumentsError('Blur kernel size cannot be greater than or equal to 25!', self.ImagePath)
        self.ImageData = cv.bilateralFilter(self.ImageData, blur_kernel_size, sigma_color, sigma_space)
        return self.ImageData

    @add_transformation('simple threshold')
    def simple_threshold(self, threshold_value1:int=127, threshold_value2:int=255) -> MatLike:
        '''
            If the pixel value is greater than the threshold, it becomes black. If less, it becomes threshold_value2.
            Arguments:
                - threshold_value1 : if average value of pixel exceeds this, it becomes black.
                - threshold_value2 : grayscale value to set pixel if less than threshold_value
        '''
        if threshold_value1 >= threshold_value2:
            raise InvalidThresholdArgumentsError('Threshold value 2 cannot be less than threshold value 1!', self.ImagePath)
        else:
            self.ImageData = cv.threshold(self.ImageData, threshold_value1, threshold_value2, cv.THRESH_BINARY)[1]
            return self.ImageData

    @add_transformation('adaptive threshold')
    def adaptive_threshold(self, color_if_less_than_threshold:int=255, kernel_size:int=31, constant:int=2) -> MatLike:
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
            self.ImageData = cv.adaptiveThreshold(self.ImageData, color_if_less_than_threshold, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, kernel_size, constant)
            return self.ImageData

    @add_transformation('otsu threshold')
    def otsu_threshold(self, threshold_value1:int=0, threshold_value2:int=255) -> MatLike:
        '''
            ! WORKS WELL WITH BIMODAL IMAGES, BUT MAY FAIL TO BINARIZE IMAGES THAT ARE NOT BIMODAL !
            Picks a threshold value that is within the peaks of a images histogram.
        '''
        if threshold_value1 >= threshold_value2:
            raise InvalidThresholdArgumentsError('Threshold value 2 cannot be less than threshold value 1!', self.ImagePath)
        self.ImageData = cv.threshold(self.ImageData, threshold_value1, threshold_value2, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        return self.ImageData

    @add_transformation('canny edge threshold')
    def canny_edge_threshold(self, threshold_value1:int=125, threshold_value2:int=175, aperture_size:int=3, l2_gradient:bool=False) -> MatLike:
        '''
            ! CAN CAUSE POOR RESULTS WITH UNEVEN LIGHTING OR EXCESS NOISE !
            Detect edges in the image.
            Arguments:
                - threshold_value1 : lower limit of the threshold
                - threshold_value2 : upper limit of the threshold
        '''
        if threshold_value2 < threshold_value1:
            raise InvalidThresholdArgumentsError('Threshold value 1 must be greater than threshold value 2!', self.ImagePath)
        else:
            self.ImageData = cv.Canny(self.ImageData, threshold_value1, threshold_value2, apertureSize=aperture_size, L2gradient=l2_gradient)
            return self.ImageData

    @add_transformation('dilate')
    @invert_colors_for_processing
    def dilate(self, kernel_size:int=3, iterations:int=1) -> MatLike:
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
            self.ImageData = cv.dilate(self.ImageData, kernel, iterations=iterations)
            return self.ImageData

    @add_transformation('erode')
    @invert_colors_for_processing
    def erode(self, kernel_size:int=3, iterations:int=1) -> MatLike:
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
            self.ImageData = cv.erode(self.ImageData, kernel, iterations=iterations)
            return self.ImageData

    @add_transformation('open pixels (erode then dilate)')
    @invert_colors_for_processing
    def open_pixels(self, kernel_size:int=3) -> MatLike:
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
            self.ImageData = cv.morphologyEx(self.ImageData, cv.MORPH_OPEN, kernel)
            return self.ImageData

    @add_transformation('close pixels (dilate then erode)')
    @invert_colors_for_processing
    def close_pixels(self, kernel_size:int=3) -> MatLike:
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
            self.ImageData = cv.morphologyEx(self.ImageData, cv.MORPH_CLOSE, kernel)
            return self.ImageData

    @add_transformation('draw contours')
    def draw_contours(self, min_threshold:int=1_000, max_threshold:int=500_000, contour_method:str='tree', contour_mode:str='simple', filter:bool=True,) -> List[List[int]]:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
        def filter_contours(image_contours:List[MatLike]) -> List[MatLike]:
            '''
                FUNCTIONDOCSTRING
                Arguments:
                    -
            '''
            filtered_contours = [contour for contour in image_contours if cv.contourArea(contour) > min_threshold and cv.contourArea(contour) < max_threshold]
            return filtered_contours
        __methods:Dict[str, Any] = {
                'external' : cv.RETR_EXTERNAL, # retrieves only external colors, disregarding any contours inside the objects
                'list' : cv.RETR_LIST, # retrives all contours without any hierarchy
                'ccomp' : cv.RETR_CCOMP, # retrieves all contours and organizes them into a two-level hierarchy. Top level contains the outer boundaries of the objects, and the second level contains the boundaries of the inner holes
                'tree' : cv.RETR_TREE # retrives all the contours and reconstructs a full hierarchy of nestted contours
        }
        __modes:Dict[str, Any] = {
                'none' : cv.CHAIN_APPROX_NONE, # stores all contours points without approximating any of them
                'simple' : cv.CHAIN_APPROX_SIMPLE, # compress horizontal, vertical, and diagonal segments and leaves only their endpoints. If a contour is straight, only the endpoints are stores
                'tc89_li' : cv.CHAIN_APPROX_TC89_L1, # variant of douglas-peucker algorithm
                'tc9_kcos' : cv.CHAIN_APPROX_TC89_KCOS  # variant of douglas-peucker algorithm
        }
        __method = __methods.get(contour_method, cv.RETR_TREE)
        __mode = __modes.get(contour_mode, cv.CHAIN_APPROX_SIMPLE)
        print(f'{__method=}', f'{__mode=}', sep='\n\n')
        contours = cv.findContours(self.ImageData, __method, __mode)[0]
        grayscale = cv.cvtColor(self.OriginalImageData, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(grayscale)
        if filter:
            potential_images = filter_contours(contours)
        else:
            potential_images = contours
        rect_info = []
        for contour in potential_images:
            x, y, w, h = cv.boundingRect(contour)
            if y < (self.ImageData.shape[0] / 5) and y > (self.ImageData.shape[0] / 12):
                continue
            else:
                r_info = [x, y, w, h]
                rect_info.append(r_info)
                cv.rectangle(self.ImageData, (x, y), (x + w, y + h), 255, -1)
            cv.bitwise_and(self.ImageData, self.ImageData, mask=mask)
        return rect_info
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
            Raised when the scale is greater than or equal to 1.0 when shrinking an image or less than or equal to 1.0 when enlarging an image.
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
            Raised when the blur kernel size is greater than 9 (or 25 in a bilateral filter) or the sigma value of a gaussian blur is greater than 3
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
            Raised when the threshold is greater than or equal to 250 or the size of neighborhood is greater than or equal to 45 in an adaptive threshold.
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
            Raised when the neighborhood size is greater than or equal to 10 when eroding or dilating.
            Arguments:
                - error_message : string that contains \\n for each new line followed by desired whitespace. Ex : "Blur kernel size cannot be greater than or equal to 10!\\nAnything greater than 10 will result in an image quality that is too poor.\\nTypically a smaller value will produce better results."
                - image_to_process_path : ImagePath property of ImageToProcess
        '''
        formatted_error_message = format_exception_new_lines(self, error_message, image_to_process_path)
        super().__init__(formatted_error_message)

class TransformationFailedError(Exception):
    '''Raised when a transformation process fails outside of an established try:except block.'''
    def __init__(self, original_error_message:str, error_message:str, image_to_process_path:ImageToProcess.ImagePath) -> None:
        '''
            Raised when a transformation process fails outside of an established try:except block.
            Arguments:
                - original_error_message : error message from the origin error (except Error as e)
                - custom_error_message : string that contains \\n for each new line followed by desired whitespace. Ex : "Blur kernel size cannot be greater than or equal to 10!\\nAnything greater than 10 will result in an image quality that is too poor.\\nTypically a smaller value will produce better results."
                - image_to_process_path : ImagePath property of ImageToProcess
        '''
        formatted_custom_error_message = format_exception_new_lines(self, error_message, image_to_process_path)
        super().__init__(original_error_message + formatted_custom_error_message)



