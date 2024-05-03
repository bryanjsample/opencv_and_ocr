from image_to_process import ImageToProcess
import cv2
import pytesseract
from pytesseract import Output

class ImageToExtract():
    ''''''
    def __init__(self, image:ImageToProcess.ImageData, image_path:ImageToProcess.ImagePath) -> None:
        self._image_path:str = image_path
        self._image_data:cv2.typing.MatLike = image

    @property
    def ImagePath(self) -> str:
        return self._image_path

    @property
    def ImageData(self) -> cv2.typing.MatLike:
        return self._image_data
    @ImageData.setter
    def ImageData(self, new_data_value:cv2.typing.MatLike):
        self._image_data = new_data_value

    def __str__(self) -> str:
        return f'\n    Image Path : {self.ImagePath}\n'

    def __repr__(self) -> str:
        return f'\n    Image Path : {self.ImagePath}\n'

    def write_boxes_of_text(self) -> None:
        d:dict = pytesseract.image_to_data(self.ImageData, output_type=Output.DICT) # sort image data into dictionary
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 60:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                self.ImageData = cv2.rectangle(self.ImageData, (x,y), (x+w, y+h), (0,255,0), 2)




def main():
    test:ImageToProcess = ImageToProcess('recipe_card.jpeg')
    test.enlarge_image_size(scale_x=2, scale_y=2)
    test.convert_to_grayscale()
    test.median_blur(blur_kernel_size=5)
    test.otsu_threshold()
    test.otsu_threshold()
    # test.dilate()
    test.erode()
    testextract = ImageToExtract(image=test.ImageData, image_path=test.ImagePath)
    testextract.write_boxes_of_text()
    cv2.imshow('all settings' , testextract.ImageData)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()

