'''
    module.py
    Author : Bryan Sample

    DESCRIPTION
'''
from exception_handling import InvertedColorError, TransformationFailedError
from typing import Callable
import cv2 as cv

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