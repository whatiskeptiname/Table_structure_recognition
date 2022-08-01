import warnings
import pandas as pd
import numpy as np
from typing import Tuple, List

from src.htmlTableParser import HTMLTable
from src.ocr import OCR


class Structure:
    '''
    Class to store table structure recognized by\n
    `StructureRecognizer.splitOnCells` and convert its format to `pandas.DataFrame`
    
    All structure is divided into individual `Cell`s 
    '''
    def __init__(self, 
                 image: np.array,
                 boundingBoxes: List[Tuple[int, int, int, int]],
                 ocr = False):
        self.image = image
        self.cells = self._initCells(boundingBoxes)
        self._initContent(ocr)
        self.HTMLCode = self._initHTMLCode()
        
    @property
    def boundingBoxes(self):
        return BoundingBoxList(cell.bbox for cell in self.cells)
    
    @property
    def gridCoords(self):
        return BoundingBoxList(cell.gridCoord for cell in self.cells)
        
    @property
    def content(self):
        return [cell.content for cell in self.cells]
        
    def _initCells(self, boundingBoxes: List[Tuple[int, int, int, int]]) -> list:
        gridCoords = self._makeGridCoords(boundingBoxes)
        return [Cell(bbox, gridCoord) for bbox, gridCoord in zip(boundingBoxes, gridCoords)]
    
    def _makeGridCoords(self, boundingBoxes: List[Tuple[int, int, int, int]]) -> list:
        bbox = np.array(boundingBoxes)
        gridCoords = list(zip(*[self._makeGridCoordsInner(bbox[:, coordIndex])
                                for coordIndex
                                in range(4)]))
        return gridCoords
        
    def _makeGridCoordsInner(self, values: np.array) -> list:
        mapDict = {value:index for index, value in enumerate(np.unique(values))}
        return [mapDict[value] for value in values]
    
    def saveContent(self, contentList: list) -> None:
        if len(contentList) != len(self.cells):
            warnings.warn(f'Cells ({len(self.cells)}) and recognized text ({len(contentList)}) amounts do not match.')
        for cell, content in zip(self.cells, contentList):
            cell.content = content
            
    def _initContent(self, ocr) -> None:
        if not ocr:
            for cell in self.cells:
                cell.makeTestContent()
            return
        ocr = OCR(self)
        ocr.recognize()
        
    def _initHTMLCode(self) -> str:
        parser = HTMLTable(self)
        return parser.getFormattedCode()
                
    def generateDataFrame(self) -> pd.DataFrame:
        '''
        Function to generate pandas DataFrame based on cells coordinates
        
        # How does it work?
        Example:\n
        
        Input structure:        
        ```python
        self.gridCoords = [(0, 0),(0, 0), -> 'name'
         (1, 0),(1, 0), -> 'AB'
         (2, 0),(3, 0), -> 'FDCS'
         (...)
         (0, 1),(0, 1), -> ''
         (1, 1),(1, 1), -> '#bcktr'
         (2, 1),(2, 1), -> '#bcktr'
         (...)
         (0, 2),(0, 2), -> 'odd_even'
         (0, 3),(0, 3), -> 'wicked_oe'
         (...)
         (1, 2),(1, 2), -> '4'
         (1, 3),(1, 3), -> '64'
         (...)
         (2, 2),(2, 2), -> '0'
         (...)]
        
        # Code in this function converts it to:
                
        [
            ['name', '', 'odd_even', 'wicked_oe', 'appendlast', ...],
            ['AB', '#bcktr', '4', '64', '43', ...],
            ['FDCS', '#bcktr', '0', '0', '24', ...],
            ['FDCS', '#Tbcktr', '0', '0', '1'],
            ...
        ]

        # or in other way

        [
            ['(0, 0),(0, 0)', '(0, 1),(0, 1)', '(0, 2),(0, 2)', ...],
            ['(1, 0),(1, 0)', '(1, 1),(1, 1)', '(1, 2),(1, 2)', ...],
            ['(2, 0),(3, 0)', '(2, 1),(2, 1)', '(2, 2),(2, 2)', ...],
            ['(2, 0),(3, 0)', '(3, 1),(3, 1)', '(3, 2),(3, 2)', ...],
            ...
        ]
                
        ```
        '''
        cells = sorted(self.cells, key=lambda cell: cell.gridCoord.rowLeft)
        dfColumns = np.array([[None 
                            for _ 
                            in range(np.max(self.gridCoords.rowLeft) + 1)] 
                            for _ 
                            in range(np.max(self.gridCoords.colTop) + 1)])
        for cell in cells:
            coord = cell.gridCoord
            rowCoords = slice(coord.rowLeft, coord.rowRight+1)
            colCoords = slice(coord.colTop, coord.colDown+1)
            dfColumns[colCoords, rowCoords] = cell
        return pd.DataFrame(dfColumns)
    
    
class Cell:
    '''
    Class to store informations about single cell in table:
    - `bbox` - bounding box with image coordinates
    - `gridCoord` - coordinates in table
    - `content` - text recognized from that cell on image
    '''
    def __init__(self,
                 boundingBox: Tuple[int, int, int, int],
                 gridCoord: Tuple[int, int, int, int]):
        self.bbox = BoundingBox(boundingBox)
        self.gridCoord = BoundingBox(gridCoord)
        self.content = None
        
    def makeTestContent(self):
        if self.gridCoord.firstCoord == self.gridCoord.secondCoord:
            self.content = str(self.gridCoord.firstCoord)
            return
        self.content = '-'.join(str(coord) for coord in self.gridCoord.coords)
        
    def __repr__(self):
        return self.content
        
class BoundingBox:
    '''
    Class to store info about bounding box coordinates and\n
    make it easier to access them
    '''
    def __init__(self,
                 boundingBox: Tuple[int, int, int, int]):
        self.rowLeft = boundingBox[0]
        self.colTop = boundingBox[1]
        self.rowRight = boundingBox[2]
        self.colDown = boundingBox[3]
        self.row = (self.rowLeft, self.rowRight)
        self.col = (self.colTop, self.colDown)
        self.firstCoord = (self.rowLeft, self.colTop)
        self.secondCoord = (self.rowRight, self.colDown)
        self.coords = [self.firstCoord, self.secondCoord]
        
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.coords[int(idx/2)][idx%2] for idx in range(*index.indices(4))]
        return self.coords[int(index/2)][index%2]
        
    def __repr__(self):
        return f'{self.firstCoord},{self.secondCoord}'
    
class BoundingBoxList(list):
    '''
    Class to make accessing `BoundingBox` fields easier
    '''   
    @property
    def rowLeft(self):
        return [item.rowLeft for item in self]
    
    @property
    def colTop(self):
        return [item.colTop for item in self]
    
    @property
    def rowRight(self):
        return [item.rowRight for item in self]
    
    @property
    def colDown(self):
        return [item.colDown for item in self]
    
    @property
    def row(self):
        return [item.row for item in self]
    
    @property
    def col(self):
        return [item.col for item in self]
    
    @property
    def firstCoord(self):
        return [item.firstCoord for item in self]
    
    @property
    def secondCoord(self):
        return [item.secondCoord for item in self]
    
    @property
    def coords(self):
        raise AttributeError('No need to use `.coords` property. `.coords` are returned by default when no property is used')
        