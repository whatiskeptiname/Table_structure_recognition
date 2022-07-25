import cv2
import numpy as np
from typing import List, Tuple, Union


def splitOnCells(image: np.array) -> List[List[Tuple[int, int]]]:
    '''
    Main module function that splits table image on separate cells.
    Recognizes table structure both with and without visual line separation between cells
    '''
    return _splitOnCells(image)[0]

def _splitOnCells(image: np.array,
                 shiftX: int = 0,
                 shiftY: int = 0):
    '''
    Recursive function to make more precise splits the deeper it goes
    '''
    
    # prepare all common stuff
    columnValues = image.sum(axis=0)
    rowValues = image.sum(axis=1)
    
    # get split points by visible split lines
    colSplitPoints, rowSplitPoints = getLineSplitPoints(
        columnValues,
        rowValues
        )
    
    # if any splitPoints found, 
    # go into recursion with splitted image 
    # and return results
    if colSplitPoints is not None:
        boundingBoxes = _findBoundingBoxes('col', image, colSplitPoints, shiftX, shiftY)
        return boundingBoxes, False
    if rowSplitPoints is not None:
        boundingBoxes = _findBoundingBoxes('row', image, rowSplitPoints, shiftX, shiftY)
        return boundingBoxes, False
    
    # get splitPoints by invisible split lines
    colEmptySplitPoints, rowEmptySplitPoints = getEmptyLineSplitPoints(
        columnValues,
        rowValues,
        image.shape
    )
    
    # if any splitPoints found
    # go into recursion with splitted image
    # and return results
    if colEmptySplitPoints is not None:
        boundingBoxes = _findBoundingBoxes('col', image, colEmptySplitPoints, shiftX, shiftY)
        return boundingBoxes, False
    if rowEmptySplitPoints is not None:
        boundingBoxes = _findBoundingBoxes('row', image, rowEmptySplitPoints, shiftX, shiftY)
        return boundingBoxes, False
    
    # return bounding box if no further splits possible
    return [
        (shiftX, shiftY),
        (shiftX + image.shape[1], shiftY + image.shape[0])
    ], True
    
def _findBoundingBoxes(splitType: str,
              image: np.array, 
              splitPoints: List[Tuple[int]], 
              shiftX: int,
              shiftY: int):
    '''
    Function to do common stuff in `_splitOnCells`
    '''
    retBoundingBoxes = []
    for splitPoint1, splitPoint2 in splitPoints:
        if splitType == 'col':
            boundingBoxes, wasLowestSplit = _splitOnCells(
                image[:, splitPoint1:splitPoint2],
                shiftX + splitPoint1,
                shiftY
            )
        elif splitType == 'row':
            boundingBoxes, wasLowestSplit = _splitOnCells(
                image[splitPoint1: splitPoint2, :],
                shiftX,
                shiftY + splitPoint1
            )
        if wasLowestSplit:
            retBoundingBoxes.append(boundingBoxes)
        else:
            retBoundingBoxes.extend(boundingBoxes)
    return retBoundingBoxes

def getLineSplitPoints(columnValues: np.array,
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
                colSplitPoints = _addBorderpoints(colSplitPoints, thresh.shape[1])
                colSplitPoints = _createPoints(colSplitPoints)
                if colSplitPoints.size > 2:
                    if returnFirstFound:
                        return colSplitPoints, None
                    colsDone = True
                
        if not rowsDone:
            rowSplitPoints = np.nonzero(~np.any(thresh, axis=1))[0]
            if rowSplitPoints.size:
                rowSplitPoints = _addBorderpoints(rowSplitPoints, thresh.shape[0])
                rowSplitPoints = _createPoints(rowSplitPoints)
                if rowSplitPoints.size > 2:
                    if returnFirstFound:
                        return None, rowSplitPoints
                    rowsDone = True
        
        if rowsDone and colsDone:
            break
        
    colSplitPoints = colSplitPoints if colsDone else None
    rowSplitPoints = rowSplitPoints if rowsDone else None
    return colSplitPoints, rowSplitPoints

def getEmptyLineSplitPoints(columnValues: np.array,
                            rowValues: np.array,
                            imageShape: tuple,
                            maxFilledInEmptyLine: int = 3,
                            minEmptyLineDistanceFromBorder: tuple = (5,5),
                            minEmptyLineWidth: tuple = (14,4),
                            maxMissingToBeStillFull = 5) -> Tuple[Union[List[int], None]]:
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
    
    isColEmpty = (columnValues > imageShape[0] - maxFilledInEmptyLine)
    isRowEmpty = (rowValues > imageShape[1] - maxFilledInEmptyLine)
    isColFull = (columnValues < maxMissingToBeStillFull)
    isRowFull = (rowValues < maxMissingToBeStillFull)
    
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
    emptyCols = _getconsecutiveEmptyLines(isColEmpty)
    emptyCols = _filterEmptyLines(emptyCols,
                                  imageShape[1],
                                  minEmptyLineDistanceFromBorder[0],
                                  minEmptyLineWidth[0])
    
    emptyRows = _getconsecutiveEmptyLines(isRowEmpty)
    emptyRows = _filterEmptyLines(emptyRows,
                                  imageShape[0],
                                  minEmptyLineDistanceFromBorder[1],
                                  minEmptyLineWidth[1])
    
    # computing mid point for every group of empty lines
    splitPointsCol = [int(start+width/2) for start, width in emptyCols]
    splitPointsRow = [int(start+width/2) for start, width in emptyRows]
    
    # adding border points to each set
    splitPointsCol = _addBorderpoints(splitPointsCol, imageShape[1]) if len(splitPointsCol) else None
    splitPointsRow = _addBorderpoints(splitPointsRow, imageShape[0]) if len(splitPointsRow) else None
    
    # creating points with `ignoreDistanceLessThan == 0`, because
    # such poinst were filtered out in `_filterEmptyLines`
    if splitPointsCol is not None:
        splitPointsCol = _createPoints(splitPointsCol, 0)
    if splitPointsRow is not None:
        splitPointsRow = _createPoints(splitPointsRow, 0)
    
    return splitPointsCol, splitPointsRow

def _getconsecutiveEmptyLines(isLineEmpty: np.array,
                              minDistanceFromFull = 3) -> List[Tuple[int, int]]:
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
            if lastEmptyIndex + minDistanceFromFull > index:
                try:
                    emptyLines.pop()
                except IndexError:
                    # pop from empty list
                    pass
            lastFullIndex = index
        elif lastValue == 1 and lastFullIndex + minDistanceFromFull < index-1:
            lastEmptyIndex = index - 1
            emptyLines.append((currentConsecutiveStartIndex, currentOccurrences))
        currentConsecutiveStartIndex = index + 1
        lastValue = item
        currentOccurrences = 1
    if lastValue == 1 and lastFullIndex + minDistanceFromFull < index-1:
        emptyLines.append((currentConsecutiveStartIndex, currentOccurrences))
    
    return emptyLines

def _filterEmptyLines(emptyLines: List[Tuple[int, int]], 
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
    
def _addBorderpoints(splitPoints: np.array, maxValue: int):
    return np.unique(np.array((
        0, 
        *splitPoints,
        maxValue
        )))
    
def _createPoints(splitPoints: np.array,
                 ignoreDistanceLessThan: int = 6):
    return np.array([(splitPoint1, splitPoint2)
                     for splitPoint1, splitPoint2 
                     in zip(splitPoints, splitPoints[1:]) 
                     if splitPoint2 - splitPoint1 > ignoreDistanceLessThan])