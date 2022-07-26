import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from src.errorHandling import checkParameters, Argument
from types import FunctionType
from typing import List, Tuple

PKey = 112
escKey = 27

validParameters = {
    'showInLoop' : [Argument('images', list),
                    Argument('nameParams', dict, nullable=True),
                    Argument('graphFunctions', [FunctionType, list], nullable=True),
                    Argument('hiddenGraphFunction', FunctionType, nullable=True)]
}


def show(image, 
         destroy: bool = True, 
         nameParams: dict = None, 
         delay: int = 0, 
         graphFunctions: list = None,
         hiddenGraphFunction: list = None,
         darkenWhitePixelsBy: int = 40):
    if nameParams is None:
        nameParams = {}
    hiddenGraphs = []
    if hiddenGraphFunction is not None:
        hiddenGraphs = _addHiddenGraphs(hiddenGraphFunction, image, darkenWhitePixelsBy)
    
    if np.unique(image).size == 2:
        image = 255*image
    if graphFunctions is not None:
        if type(graphFunctions) is not list:
            graphFunctions = [graphFunctions]
        image = _addGraphs(graphFunctions, image)
        
    image = _darkenTheseWhitePixels(image, 30, darkenWhitePixelsBy)
    cv2.imshow('image', image)
    mouseCallbackParameters = {
        'nameParams' : nameParams,
        'images' : [(image, (0,0))] + hiddenGraphs,
        'nextImageIndex' : 1 % (len(hiddenGraphs) + 1)
    }
    cv2.setMouseCallback('image', _mouseEvent, mouseCallbackParameters)
    if nameParams is not None:
        cv2.setWindowTitle('image', _createImgNameFromDict(nameParams))
    pressedKey = cv2.waitKey(delay)
    if not destroy:
        return pressedKey
    cv2.destroyAllWindows()
    
def _mouseEvent(event, x, y, flags, params):
    images = params['images']
    nextImageIndex = params['nextImageIndex']
    
    color = _getColor(images, (nextImageIndex-1) % len(images), y, x)

    if event == cv2.EVENT_MOUSEMOVE:
        newImgName = params['nameParams'].copy()
        newImgName.update({
            'coords': (y,x),
            'color': color
            })
        cv2.setWindowTitle('image', _createImgNameFromDict(newImgName))
        
    if event == cv2.EVENT_LBUTTONDOWN:
        nextImage, startCoords = images[nextImageIndex]
        tempImage = images[0][0].copy()
        tempImage[
            startCoords[0]:startCoords[0] + nextImage.shape[0],
            startCoords[1]:startCoords[1] + nextImage.shape[1]
            ] = nextImage
        cv2.imshow('image', tempImage)
        params['nextImageIndex'] = (nextImageIndex + 1) % len(images)

def _createImgNameFromDict(dict):
    atFirst = ['index', 'coords', 'color']
    nameElements = [f'{item}: {dict.get(item)}' for item in atFirst if dict.get(item) is not None]
    nameElements.extend([f'{key}: {value}' for key, value in dict.items() if key not in atFirst])
    return '; '.join(nameElements)
    # return '; '.join(f'{key}: {value}' for key, value in dict.items() if value is not None)

def _darkenTheseWhitePixels(image: np.array, below: int,  by: int = 40):
    if not by:
        return image
    return cv2.threshold(image, below, 255-by, cv2.THRESH_BINARY)[1]
    
def _getColor(images, currentImageIndex, y, x):
    # edge case when image is so small but window needs to be bigger to fit in buttons
    # and x, y are out of image.shape
    color = 0
    try:
        color = images[0][0][y,x]
        color = images[0][0][y,x,0]
        color = images[currentImageIndex][0][y,x]
        color = images[currentImageIndex][0][y,x,0]
    except IndexError:
        pass
    return color

@checkParameters(validParameters)
def showInLoop(images: list, 
               delay: int = 0, 
               nameParams: dict = None, 
               graphFunctions = None,
               hiddenGraphFunction = None,
               darkenWhitePixelsBy: int = 40):
    '''
    Function to enable showing many images with graphs\n
    and hidden graphs available on clik in a loop.\n
    By default shows cursor position and related pixel color in a title field
    
    # Parameters:
    - images: `list`
    - delay: `int`, default `0`
        - delay in ms between automatically displaying next images, \n
        `0` means no auto scrolling
    - nameParams: `dict`
        - params to build image name; rarely useful
    - graphFunctions: `list`
        - list of functions that create graphs based on image and insert it into that image
    - hiddedGraphFunction: `function`
        - function to create images that alter displayed image after clicking on it
    - darkenWhitePixelsBy: `int`
    
    # Note:
    - press 'p' to display previous image
    - press 'esc' to close
    - press any other key to display next image
    - click on image to display hidden images created by `hiddenGraphFunction`
    
    # Usage example:
    
    ```python
    showInLoop(images, 
               graphFunctions=addPlots,
               hiddenGraphFunction=addHiddenPlots)
               
    # that shows images in loop, each with associated graph and skeleton that
    # shows after clicking on image
    ```
    '''
    if type(graphFunctions) is not list and graphFunctions is not None:
        graphFunctions = [graphFunctions]
    if nameParams is None:
        nameParams = {}
    nameParams = _prepareNameParams(nameParams, len(images))
    nImages = len(images)
    while True:
        index = 0
        while index < nImages:
            image = images[index]
            try:
                currentNameParams = dict([(key, value[index]) for key, value in nameParams.items()])
            except IndexError as e:
                # list index out of range
                cv2.destroyAllWindows()
                raise IndexError('unequal amount of images and name params') from None
            pressedKey = show(image,
                                destroy = False,
                                nameParams = currentNameParams,
                                delay = delay,
                                graphFunctions = graphFunctions,
                                hiddenGraphFunction = hiddenGraphFunction,
                                darkenWhitePixelsBy = darkenWhitePixelsBy)
            if pressedKey == 112:
                index = (index-1)%nImages
                continue
            if pressedKey == 27:
                cv2.destroyAllWindows()
                return
            index += 1
            
def _prepareNameParams(nameParams, nImages):
    for key, value in nameParams.items():
        if not isinstance(value, list):
            nameParams[key] = [value] * nImages
    if nameParams.get('index') is None:
        nameParams['index'] = range(nImages)
    return nameParams
            
            
def _addGraphs(graphFunctions: list, image: np.array) -> np.array:
    for graphFunction in graphFunctions:
        image = graphFunction(image)
    return image
            
def _addHiddenGraphs(hiddenGraphFunction: list, image: np.array, darkenWhitePixelsBy: int) -> list:
    try:
        return hiddenGraphFunction(image)
    except TypeError:
        # wrong amount of parameters
        return hiddenGraphFunction(image, darkenWhitePixelsBy)

def addPlots(image: np.array) -> np.array:
    '''
    Function to add vertical and horizontal plots to image\n
    showing aggregated vertical and horizontal pixels sums
    '''
    linesSums = [image.sum(axis=axis)/255 for axis in [0,1]]
    plots = []
    for axis in [0,1]:
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.axis('off')
        ax.margins(0)
        fig.tight_layout(pad=0)

        ax.plot(linesSums[axis], 'r')
        # parameter `'r'` means plot with red line, but matplotlib outputs
        # rgb and openCV writes it as bgr so final color is blue

        fig.canvas.draw()
        plotImage = np.frombuffer(fig.canvas.tostring_rgb(), dtype = 'uint8')
        plotImage = plotImage.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if axis == 0:
            dsize = (image.shape[1], plotImage.shape[0])
        else:
            plotImage = plotImage.transpose(1,0,2)
            dsize = (plotImage.shape[1], image.shape[0])
        plotImage = cv2.resize(plotImage, dsize)
        plots.append(plotImage)
    plots[1] = np.append(plots[1], 
                         np.zeros((plots[0].shape[0], plots[1].shape[1], 3), dtype='uint8'),
                         axis=0)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    retImage = np.append(
        image, 
        plots[0], 
        axis=0)
    retImage = np.append(
        retImage, 
        plots[1], 
        axis=1)
    return retImage

def addHiddenPlots(image: np.array, darkenWhitePixelsBy: int) -> tuple:
    '''
    Used to show table skeleton after clicking on displayed image
    '''
    columnValues = image.sum(axis=0)
    rowValues = image.sum(axis=1)
    kindOfSkeleton = np.matmul(rowValues[:, None], columnValues[None, :])
    kindOfSkeleton = (255*kindOfSkeleton/np.max(kindOfSkeleton)).astype('uint8')
    kindOfSkeleton = cv2.cvtColor(kindOfSkeleton, cv2.COLOR_GRAY2BGR)
    thresholded = [(_darkenTheseWhitePixels(kindOfSkeleton, value, darkenWhitePixelsBy), (0,0)) for value in range(0,50,10)]
    return [(kindOfSkeleton, (0,0))] + thresholded


def showBboxOnImage(image: np.array, 
                    bboxList: List[Tuple[int, int]],
                    cumulative: bool = False, 
                    **kwargs):
    if cumulative:
        bboxImage = image.copy()
        for index, bbox in enumerate(bboxList):
            bboxImage = cv2.rectangle(bboxImage, bbox[:2], bbox[2:], (0,0,0), 2)
        hiddenImage = 255*image if kwargs.get('graphFunctions') is None else cv2.cvtColor(255*image, cv2.COLOR_GRAY2BGR)
        
        show(215*bboxImage, 
             hiddenGraphFunction = lambda img: [(hiddenImage, (0,0))],
             **kwargs)
        return
        
    if kwargs.get('nameParams') is None:
        kwargs['nameParams'] = {}
    if kwargs['nameParams'].get('bbox') is None:
        kwargs['nameParams']['bbox'] = bboxList
    showInLoop([cv2.rectangle(215*image.copy(), bbox[:2], bbox[2:], (0,0,0), 2)
                for bbox 
                in bboxList],
               **kwargs)