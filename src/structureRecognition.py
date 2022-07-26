import cv2
import numpy as np
from typing import List, Tuple, Union


class StructureRecognizer:
    '''
        Class to provide methods to perform table structure recognition
        
        ---
        # Parameters:
        - maxFilledInEmptyLine: `int`
            - max amount of black pixels in particular line\n
            to be classified as empty line
        - maxMissingToBeStillFull: `int`
            - max amount of white pixels in particular line\n
            to be classified as full line
        - minEmptyLineWidth: `tuple`
            - min distance between words in the same cell\n
                first element  -> (column split)\n
                second element -> (row split)\n
        - minEmptyLineDistanceFromBorder: `tuple`
            - min distance from cell border to bunch of empty lines\n
            to make next split
                first element  -> (column split)\n
                second element -> (row split)\n
        - minDistanceFromFull: `int`
            - min distance from full line to bunch of empty lines\n
            to make next split.\n
            This differs from previous parameter in that it uses\n
            lines detected by `maxMissingToBeStillFull` and not only cell border lines
        - ignoreSplitDistanceLessThan: `int`
            - when two split points at this or less distance from each other\n
            are detected, area in between them is ignored and\n
            not treated as another cell
            
        ---
        # Usage:
        
        ```python
        recognizer = StructureRecognizer(<parameters here>)
        cellBoundingBoxes = recognizer.splitOnCells(image)
        
        ```
            
    '''
    def __init__(self,
                 maxFilledInEmptyLine: int = 3,
                 maxMissingToBeStillFull: int = 5,
                 minEmptyLineWidth: tuple = (14,4),
                 minEmptyLineDistanceFromBorder: tuple = (5,5),
                 minDistanceFromFull: int = 3,
                 ignoreSplitDistanceLessThan: int = 6):
        self.maxFilledInEmptyLine = maxFilledInEmptyLine
        self.maxMissingToBeStillFull = maxMissingToBeStillFull
        self.minEmptyLineWidth = minEmptyLineWidth
        self.minEmptyLineDistanceFromBorder = minEmptyLineDistanceFromBorder
        self.minDistanceFromFull = minDistanceFromFull
        self.ignoreSplitDistanceLessThan = ignoreSplitDistanceLessThan

    def splitOnCells(self, image: np.array) -> List[Tuple[int, int, int, int]]:
        '''
        Main module function that splits table image on separate cells.
        Recognizes table structure both with and without visual line separation between cells
        '''
        splittedCells = self._splitOnCells(image)[0]
        unifiedCells = self._unifyBoundingBox(splittedCells)
        return unifiedCells

    def _splitOnCells(self,
                      image: np.array,
                      shiftX: int = 0,
                      shiftY: int = 0) -> List[Tuple[int, int, int, int]]:
        '''
        Recursive function to make more precise splits the deeper it goes
        '''
        
        # prepare all common stuff
        columnValues = image.sum(axis=0)
        rowValues = image.sum(axis=1)
        
        # get split points by visible split lines
        colSplitPoints, rowSplitPoints = self.getLineSplitPoints(
            columnValues,
            rowValues
            )
        
        # if any splitPoints found, 
        # go into recursion with splitted image 
        # and return results
        if colSplitPoints is not None:
            boundingBoxes = self._findBoundingBoxes('col', image, colSplitPoints, shiftX, shiftY)
            return boundingBoxes, False
        if rowSplitPoints is not None:
            boundingBoxes = self._findBoundingBoxes('row', image, rowSplitPoints, shiftX, shiftY)
            return boundingBoxes, False
        
        # get splitPoints by invisible split lines
        colEmptySplitPoints, rowEmptySplitPoints = self.getEmptyLineSplitPoints(
            columnValues,
            rowValues,
            image.shape
        )
        
        # if any splitPoints found
        # go into recursion with splitted image
        # and return results
        if colEmptySplitPoints is not None:
            boundingBoxes = self._findBoundingBoxes('col', image, colEmptySplitPoints, shiftX, shiftY)
            return boundingBoxes, False
        if rowEmptySplitPoints is not None:
            boundingBoxes = self._findBoundingBoxes('row', image, rowEmptySplitPoints, shiftX, shiftY)
            return boundingBoxes, False
        
        # return bounding box if no further splits possible
        return (
            shiftX, shiftY,
            shiftX + image.shape[1], shiftY + image.shape[0]
        ), True
        
    def _findBoundingBoxes(self,
                           splitType: str,
                           image: np.array, 
                           splitPoints: List[Tuple[int]], 
                           shiftX: int,
                           shiftY: int) -> List[Tuple[int, int, int, int]]:
        '''
        Function to do common stuff in `_splitOnCells`
        '''
        retBoundingBoxes = []
        for splitPoint1, splitPoint2 in splitPoints:
            if splitType == 'col':
                boundingBoxes, wasLowestSplit = self._splitOnCells(
                    image[:, splitPoint1:splitPoint2],
                    shiftX + splitPoint1,
                    shiftY
                )
            elif splitType == 'row':
                boundingBoxes, wasLowestSplit = self._splitOnCells(
                    image[splitPoint1: splitPoint2, :],
                    shiftX,
                    shiftY + splitPoint1
                )
            if wasLowestSplit:
                retBoundingBoxes.append(boundingBoxes)
            else:
                retBoundingBoxes.extend(boundingBoxes)
        return retBoundingBoxes

    def getLineSplitPoints(self,
                           columnValues: np.array,
                           rowValues: np.array,
                           returnFirstFound = True) -> Tuple[Union[List[int], None]]:
        '''
        # Returns `tuple` with 2 items:
            - columns split points (each points as single `int`)\n
            or `None` if not found such points
            - rows split points (each points as single `int`)\n
            or `None` if not found such points
        '''
        colsDone = False
        rowsDone = False
        kindOfSkeleton = np.matmul(rowValues[:, None], columnValues[None, :])
        kindOfSkeleton = (255*kindOfSkeleton/np.max(kindOfSkeleton)).astype('uint8')
        for value in range(40):
            thresh = cv2.threshold(kindOfSkeleton, value, 1, cv2.THRESH_BINARY)[1]
            if not colsDone:
                colSplitPoints = np.nonzero(~np.any(thresh, axis=0))[0]
                if colSplitPoints.size:
                    colSplitPoints = self._addBorderpoints(colSplitPoints, thresh.shape[1])
                    colSplitPoints = self._createPoints(colSplitPoints)
                    if colSplitPoints.size > 2:
                        if returnFirstFound:
                            return colSplitPoints, None
                        colsDone = True
                    
            if not rowsDone:
                rowSplitPoints = np.nonzero(~np.any(thresh, axis=1))[0]
                if rowSplitPoints.size:
                    rowSplitPoints = self._addBorderpoints(rowSplitPoints, thresh.shape[0])
                    rowSplitPoints = self._createPoints(rowSplitPoints)
                    if rowSplitPoints.size > 2:
                        if returnFirstFound:
                            return None, rowSplitPoints
                        rowsDone = True
            
            if rowsDone and colsDone:
                break
            
        colSplitPoints = colSplitPoints if colsDone else None
        rowSplitPoints = rowSplitPoints if rowsDone else None
        return colSplitPoints, rowSplitPoints

    def getEmptyLineSplitPoints(self,
                                columnValues: np.array,
                                rowValues: np.array,
                                imageShape: tuple) -> Tuple[Union[List[int], None]]:
        '''
        Function to find table split points based on empty lines
        
        # Parameters:
        - image: `np.array`
        - maxFilledInEmptyLine: `int`
        - minEmptyLineDistanceFromBorder: `tuple`
            - `tuple` with 2 values: 
                - first one for column splits
                - second one for row splits
        - minEmptyLineWidth: `tuple`
            - `tuple` with 2 values: 
                - first one for column splits
                - second one for row splits
                
        # Returns `tuple` with 2 items:
            - columns split points (each points as single `int`)\n
            or `None` if not found such points
            - rows split points (each points as single `int`)\n
            or `None` if not found such points
        '''
        # finding empty columns, is...Empty are boolean vectors
        
        isColEmpty = (columnValues > imageShape[0] - self.maxFilledInEmptyLine)
        isRowEmpty = (rowValues > imageShape[1] - self.maxFilledInEmptyLine)
        isColFull = (columnValues < self.maxMissingToBeStillFull)
        isRowFull = (rowValues < self.maxMissingToBeStillFull)
        
        # combining above results to be able to detect when group of empty lines
        # is just near to the drawn visual line and is not separate split point
        isColEmpty = isColEmpty*1 - isColFull*1
        isRowEmpty = isRowEmpty*1 - isRowFull*1
        # now these vectors has 3 possible values:
        # -1 -> full line
        #  0 -> not full not empty
        #  1 -> empty line
        
        # turning vectors into list of tuples with:
        # - empty lines start point (`int`)
        # - empty lines width (`int`)
        emptyCols = self._getconsecutiveEmptyLines(isColEmpty)
        emptyCols = self._filterEmptyLines(emptyCols,
                                    imageShape[1],
                                    self.minEmptyLineDistanceFromBorder[0],
                                    self.minEmptyLineWidth[0])
        
        emptyRows = self._getconsecutiveEmptyLines(isRowEmpty)
        emptyRows = self._filterEmptyLines(emptyRows,
                                    imageShape[0],
                                    self.minEmptyLineDistanceFromBorder[1],
                                    self.minEmptyLineWidth[1])
        
        # computing mid point for every group of empty lines
        splitPointsCol = [int(start+width/2) for start, width in emptyCols]
        splitPointsRow = [int(start+width/2) for start, width in emptyRows]
        
        # adding border points to each set
        splitPointsCol = self._addBorderpoints(splitPointsCol, imageShape[1]) if len(splitPointsCol) else None
        splitPointsRow = self._addBorderpoints(splitPointsRow, imageShape[0]) if len(splitPointsRow) else None
        
        # creating points with `ignoreSplitDistanceLessThan == 0`, because
        # such poinst were filtered out in `_filterEmptyLines`
        if splitPointsCol is not None:
            splitPointsCol = self._createPoints(splitPointsCol, 0)
        if splitPointsRow is not None:
            splitPointsRow = self._createPoints(splitPointsRow, 0)
        
        return splitPointsCol, splitPointsRow

    def _getconsecutiveEmptyLines(self,
                                  isLineEmpty: np.array) -> List[Tuple[int, int]]:
        '''
        Returns `list` of `tuples`\n
        Each `tuple` has 2 fields meaning:
        1. empty line start index
        2. empty line width
        '''
        emptyLines = []
        lastValue = isLineEmpty[0]
        lastFullIndex = 0 if lastValue == -1 else -np.inf
        lastEmptyIndex = 0 if lastValue == 1 else -np.inf
        currentOccurrences = 1
        currentConsecutiveStartIndex = 0

        for index, item in enumerate(isLineEmpty[1:]):
            if item == lastValue:
                currentOccurrences += 1
                continue
            if item == -1:
                if lastEmptyIndex + self.minDistanceFromFull > index:
                    try:
                        emptyLines.pop()
                    except IndexError:
                        # pop from empty list
                        pass
                lastFullIndex = index
            elif lastValue == 1 and lastFullIndex + self.minDistanceFromFull < index-1:
                lastEmptyIndex = index - 1
                emptyLines.append((currentConsecutiveStartIndex, currentOccurrences))
            currentConsecutiveStartIndex = index + 1
            lastValue = item
            currentOccurrences = 1
        if lastValue == 1 and lastFullIndex + self.minDistanceFromFull < index-1:
            emptyLines.append((currentConsecutiveStartIndex, currentOccurrences))
        
        return emptyLines

    def _filterEmptyLines(self,
                          emptyLines: List[Tuple[int, int]], 
                          totalImageWidth: int, 
                          minDistanceFromBorder: int, 
                          minWidth: int) -> List[Tuple[int, int]]:
        return [(startPoint, width)
                for startPoint, width
                in emptyLines
                if width > minWidth
                    and startPoint > minDistanceFromBorder 
                    and startPoint+width < totalImageWidth - minDistanceFromBorder
            ]
        
    def _addBorderpoints(self, 
                         splitPoints: np.array,
                         maxValue: int) -> np.array:
        return np.unique(np.array((
            0, 
            *splitPoints,
            maxValue
            )))
        
    def _createPoints(self,
                      splitPoints: np.array,
                      ignoreSplitDistanceLessThan: int = None) -> np.array:
        ignoreSplitDistanceLessThan = ignoreSplitDistanceLessThan if ignoreSplitDistanceLessThan is not None else self.ignoreSplitDistanceLessThan
        return np.array([(splitPoint1, splitPoint2)
                        for splitPoint1, splitPoint2 
                        in zip(splitPoints, splitPoints[1:]) 
                        if splitPoint2 - splitPoint1 > ignoreSplitDistanceLessThan])
        
        
    def _unifyBoundingBox(self, 
                          bbox: List[Tuple[int, int, int, int]],
                          maxDifferenceInOneGroup: int = 5):
        '''
        Reduces minor differences between bounding box coordinates\n
        returned by `_splitOnCells` separately on x and y axis:
        Example:
        ```python
        [[3,   3,  135, 28],     ->    [[3,   3,  135, 28],
        [135, 4,  226, 28],   maps to  [135, 3,  225, 28],
        [225, 29, 427, 28],     ->     [225, 28, 427, 28],
                ...                            ...
        ```
        Even though this function process each column separately,\n
        if input values come from `_splitOnCells`,\n
        it is guaranteed that (0th, 2nd) and (1st, 3rd) column will have\n
        exactly the same pool of values used due to way of splitting\n
        used in `_splitOnCells` (splitting is made in `_findBoundingBoxes`)
        
        '''
        bbox = np.array(bbox)
        bbox = list(zip(*[self._unifyBoundingBoxInner(bbox[:, coordIndex],
                                                      maxDifferenceInOneGroup)
                    for coordIndex
                    in range(4)]))
        return bbox
            
    def _unifyBoundingBoxInner(self,
                               values: np.array,
                               maxDifferenceInOneGroup: int):
        '''
        Function to group nearby bounding box coordinates\n
        '''
        prevVal = -float('inf')
        nearbyValues = []
        newGroup = True
        for currentVal in np.sort(np.unique(values)):
            if currentVal - prevVal > maxDifferenceInOneGroup:
                nearbyValues.append([currentVal])
                prevVal = currentVal
                newGroup = True
                continue
            if newGroup:
                nearbyValues.pop()
                nearbyValues.append([prevVal, currentVal])
                newGroup = False
            else:
                nearbyValues[-1].extend([currentVal])
        nearbyValues = self._mapNewBoundingBoxValues(values, nearbyValues)
        return nearbyValues

    def _mapNewBoundingBoxValues(self,
                                 originalValues: np.array,
                                 newValues: list):
        '''
        Function to map original bounding box values to corrected ones
        '''
        retDict = {}
        for row in newValues:
            rowMean = np.mean(row).astype(int)
            for value in row:
                retDict[value] = rowMean
        return [retDict[item] for item in originalValues]
    
    
    