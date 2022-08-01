from typing import Union
import cv2
import numpy as np
import pytesseract

class OCR:
    def __init__(self, 
                 structure, # of type Structure, but not imported here
                 scaleFactor: Union[int, float] = 4,
                 dilationKernelSize: int = 3):
        self.structure = structure
        self.scaleFactor = scaleFactor
        self.dilationKernelSize = dilationKernelSize
        self.cells = self._prepareCells()
        
    def _prepareCells(self) -> list:
        image = self._preprocessImage()
        parts = []
        for col, row in zip(self.structure.boundingBoxes.col, self.structure.boundingBoxes.row):
            part = image[slice(*col), slice(*row)]
            part = cv2.resize(part, (0,0), fx=self.scaleFactor, fy=self.scaleFactor)
            parts.append(part)
        return parts
    
    def _preprocessImage(self) -> np.array:
        # basicly no need to do much preprocessing as long as images shows
        # tables created only by computer software
        
        image = self.structure.image.copy()
        return image
    
    def recognize(self) -> None:
        # TODO: make it faster by merging cells
        config = '--psm 7'
        recognizedContent = [pytesseract.image_to_string(255*part, config=config).strip() for part in self.cells]
        self.structure.saveContent(recognizedContent)