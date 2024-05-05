'''
    module.py
    Author : Bryan Sample

    DESCRIPTION
'''

# error handling definitions

def format_exception_new_lines(exception_class:object, error_message:str, image_to_process_path:str) -> str:
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
    def __init__(self, error_message:str, image_to_process_path:str) -> None:
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
    def __init__(self, error_message:str, image_to_process_path:str, kernel_size_error:bool=False) -> None:
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
    def __init__(self, error_message:str, image_to_process_path:str) -> None:
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
    def __init__(self, error_message:str, image_to_process_path:str) -> None:
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
    def __init__(self, original_error_message:str, error_message:str, image_to_process_path:str) -> None:
        '''
            Raised when a transformation process fails outside of an established try:except block.
            Arguments:
                - original_error_message : error message from the origin error (except Error as e)
                - custom_error_message : string that contains \\n for each new line followed by desired whitespace. Ex : "Blur kernel size cannot be greater than or equal to 10!\\nAnything greater than 10 will result in an image quality that is too poor.\\nTypically a smaller value will produce better results."
                - image_to_process_path : ImagePath property of ImageToProcess
        '''
        formatted_custom_error_message = format_exception_new_lines(self, error_message, image_to_process_path)
        super().__init__(original_error_message + formatted_custom_error_message)

class InvertedColorError(Exception):
    ''''''
    def __init__(self, error_message:str, image_to_process_path:str) -> None:
        '''
            FUNCTIONDOCSTRING
            Arguments:
                -
        '''
        formatted_error_message = format_exception_new_lines(self, error_message, image_to_process_path)
        super().__init__(formatted_error_message)