'''
    image_to_process.py
    Author : Bryan Sample

    Class definition of ImageToProcess, which enables processing of an image to improve it's readability for Tesseract OCR along with it's decorator functions.
'''

from exception_handling import TransformationFailedError, InvalidScaleArgumentsError, InvalidBlurArgumentsError, InvalidThresholdArgumentsError, InvalidDilateOrErodeArguments, InvertedColorError, GrayscaleError, InvalidKernelSizeError
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
        if image_object.InvertedColor is True:
            pass
        elif image_object.InvertedColor is False:
            image_object.ImageData = cv.bitwise_not(image_object.ImageData)
            image_object.InvertedColor = True
        else:
            raise InvertedColorError(error_message='Something has gone wrong with the color inversion decorator. Run debugger for further information.', image_to_process_path=image_object.ImagePath)
        processing_func(image_object, *args, **kwargs)
        if image_object.CannyData is True or image_object.LaplacianData is True:
            pass
        elif image_object.CannyData is False and image_object.LaplacianData is False:
            image_object.ImageData = cv.bitwise_not(image_object.ImageData)
            image_object.InvertedColor = False
        else:
            raise InvertedColorError(error_message='Something has gone wrong with the color inversion decorator. Run debugger for further information.', image_to_process_path=image_object.ImagePath)
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
        self._scale_x:float = 1.0
        self._scale_y:float = 1.0
        self._grayscale:bool = False
        self._inverted_color:bool = False
        self._canny_data:bool = False
        self._laplacian_data:bool = False
        self._transformations:Dict[int, Tuple[str, List[Tuple[Any]|Dict[str, Any]]]]|None = None
        self._mask_rectangles:List[List[int]]|None = None
        self._mask_rect_scale:Tuple[float] = (1.0, 1.0)

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
    def ScaleX(self) -> float:
        return self._scale_x
    @ScaleX.setter
    def ScaleX(self, new_scale:float) -> None:
        self._scale_x = new_scale

    @property
    def ScaleY(self) -> float:
        return self._scale_y
    @ScaleY.setter
    def ScaleY(self, new_scale:float) -> None:
        self._scale_y = new_scale

    @property
    def Grayscale(self) -> bool:
        return self._grayscale
    @Grayscale.setter
    def Grayscale(self, bool_value:bool) -> None:
        self._grayscale = bool_value

    @property
    def InvertedColor(self) -> bool:
        return self._inverted_color
    @InvertedColor.setter
    def InvertedColor(self, bool_value:bool) -> None:
        self._inverted_color = bool_value

    @property
    def CannyData(self) -> bool:
        return self._canny_data
    @CannyData.setter
    def CannyData(self, bool_value:bool) -> None:
        self._canny_data = bool_value

    @property
    def LaplacianData(self) -> bool:
        return self._laplacian_data
    @LaplacianData.setter
    def LaplacianData(self, bool_value:bool) -> None:
        self._laplacian_data = bool_value

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
            return [kwarg_str if kwarg_str != '' else '' for kwarg_str in formatted_kwargs]
        def format_args(args_list:List[Tuple|str]) -> List[str]:
            formatted_args:List[str] = []
            for args in args_list:
                if args == 'None':
                    formatted_args.append('')
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

    @property
    def MaskRectInfo(self) -> List[List[int]]|None:
        return self._mask_rectangles
    @MaskRectInfo.setter
    def MaskRectInfo(self, rect_info:List[List[int]]) -> None:
        self._mask_rectangles = rect_info

    @property
    def MaskRectScale(self) -> Tuple[float]:
        return self._mask_rect_scale
    @MaskRectScale.setter
    def MaskRectScale(self, new_scale:Tuple[float]) -> None:
        self._mask_rect_scale = new_scale

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

    def reset_image_data(self, retain_transformations:bool=False) -> None:
        self._image_data[0] = cv.imread(self.ImagePath)
        self.ImageData = cv.imread(self.ImagePath)
        self.ScaleX = 1.0
        self.ScaleY = 1.0
        self.Grayscale = False
        self.InvertedColor = False
        self.CannyData = False
        self.LaplacianData = False
        if not retain_transformations:
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
            self.ScaleX, self.ScaleY = scale_x, scale_y
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
            self.ScaleX, self.ScaleY = scale_x, scale_y
            return self.ImageData

    @add_transformation('convert to grayscale')
    def convert_to_grayscale(self) -> MatLike:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
        if self.Grayscale is True:
            raise GrayscaleError('Color is already grayscale. No need to convert.', self.ImagePath)
        elif self.Grayscale is False:
            self.ImageData = cv.cvtColor(self.ImageData, cv.COLOR_BGR2GRAY)
            self.Grayscale = True
        else:
            raise GrayscaleError('Something went wrong with the grayscale error. Figure that out')
        return self.ImageData

    @add_transformation('inverted colors')
    def invert_color(self) -> MatLike:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
        self.ImageData = cv.bitwise_not(self.ImageData)
        self.InvertedColor = not self.InvertedColor
        self.CannyData = False
        self.LaplacianData = False
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
            raise InvalidKernelSizeError('Blur kernel size must be odd!', self.ImagePath)
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
            raise InvalidKernelSizeError('Blur kernel size must be odd!', self.ImagePath)
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
            raise InvalidKernelSizeError('Blur kernel size must be odd!', self.ImagePath)
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
            raise InvalidKernelSizeError('Blur kernel size must be odd!', self.ImagePath)
        elif blur_kernel_size > 25:
            raise InvalidBlurArgumentsError('Blur kernel size cannot be greater than or equal to 25!', self.ImagePath)
        self.ImageData = cv.bilateralFilter(self.ImageData, blur_kernel_size, sigma_color, sigma_space)
        return self.ImageData

    @add_transformation('simple threshold')
    def simple_threshold(self, threshold_value:int=127, color_value_if_less_than_threshold:int=255) -> MatLike:
        '''
            If the pixel value is greater than the threshold, it becomes black. If less, it becomes threshold_value2.
            Arguments:
                - threshold_value : if average value of pixel exceeds this, it becomes black.
                - color_value_if_less_than_threshold : grayscale value to set pixel if less than threshold_value
        '''
        self.ImageData = cv.threshold(self.ImageData, threshold_value, color_value_if_less_than_threshold, cv.THRESH_BINARY)[1]
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
            raise InvalidKernelSizeError('Kernel size must be odd!', self.ImagePath)
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
            self.InvertedColor = True
            self.CannyData = True
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
            raise InvalidKernelSizeError('Kernel size must be odd!', self.ImagePath)
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
            raise InvalidKernelSizeError('Kernel size must be odd!', self.ImagePath)
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
            raise InvalidKernelSizeError('Kernel size must be odd!', self.ImagePath)
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
            raise InvalidKernelSizeError('Kernel size must be odd!', self.ImagePath)
        elif kernel_size > 9:
            raise InvalidDilateOrErodeArguments('Size of neighborhood cannot be greater than 9!', self.ImagePath)
        else:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            self.ImageData = cv.morphologyEx(self.ImageData, cv.MORPH_CLOSE, kernel)
            return self.ImageData
        
    @add_transformation('laplacian filterting')
    def laplacian_filter(self, kernel_size:int=3, desired_depth:int=cv.CV_64F, laplacian_scale:float=1.0, delta:int=0, border_type:str='default') -> MatLike:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
        _border_types:dict[str, Any] = {
                'default' : cv.BORDER_DEFAULT,
                'constant' : cv.BORDER_CONSTANT,
                'isolated' : cv.BORDER_ISOLATED,
                'replicate' : cv.BORDER_REPLICATE,
                'reflect' : cv.BORDER_REFLECT,
                'wrap' : cv.BORDER_WRAP,
                'reflect_101' : cv.BORDER_REFLECT_101,
                'transparent' : cv.BORDER_TRANSPARENT,
        }
        if kernel_size % 2 != 1:
            raise InvalidKernelSizeError('Kernel size must be odd!', self.ImagePath)
        border_value = _border_types.get(border_type, cv.BORDER_DEFAULT)
        self.ImageData = cv.Laplacian(src=self.ImageData, ddepth=desired_depth, ksize=kernel_size, scale=laplacian_scale, delta=delta, borderType=border_value)
        self.ImageData = np.uint8(self.ImageData)
        self.InvertedColor = True
        self.LaplacianData = True
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
        print(f'{_method=}', f'{_mode=}', sep='\n\n')
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
            if y < (self.ImageData.shape[0] / 5) and y > (self.ImageData.shape[0] / 12):
                continue
            else:
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
