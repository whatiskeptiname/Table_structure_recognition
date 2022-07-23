import cv2
from cv2 import threshold
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from src.errorHandling import checkParameters, Argument
from types import FunctionType

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
         darkenWhitePixelsBy: int = 60):
    if nameParams is None:
        nameParams = {}
    hiddenGraphs = []
    if hiddenGraphFunction is not None:
        hiddenGraphs = _addHiddenGraphs(hiddenGraphFunction, image, darkenWhitePixelsBy)
    if np.unique(image).size == 2:
        image = 255*image
    if graphFunctions is not None:
        image = _addGraphs(graphFunctions, image)
        
    image = _darkenTheseWhitePixels(image, darkenWhitePixelsBy)
    cv2.imshow('image', image)
    mouseCallbackParameters = {
        'image' : image,
        'nameParams' : nameParams,
        'changeableImages' : [(image, (0,0))] + hiddenGraphs,
        'nextChangeableIndex' : 1 % len(hiddenGraphs)
    }
    cv2.setMouseCallback('image', _mouseEvent, mouseCallbackParameters)
    if nameParams is not None:
        cv2.setWindowTitle('image', _createImgNameFromDict(nameParams))
    pressedKey = cv2.waitKey(delay)
    if not destroy:
        return pressedKey
    cv2.destroyAllWindows()
    
def _mouseEvent(event, x, y, flags, params):
    image = params['image']
    try:
        color = image[y,x,0]
    except IndexError:
        color = image[y,x]

    if event == cv2.EVENT_MOUSEMOVE:
        newImgName = params['nameParams'].copy()
        newImgName.update({
            'coords': (y,x),
            'color': color
            })
        cv2.setWindowTitle('image', _createImgNameFromDict(newImgName))
        
    if event == cv2.EVENT_LBUTTONDOWN:
        changeableImages = params['changeableImages']
        nextChangeableIndex = params['nextChangeableIndex']
        changeableImage, startCoords = changeableImages[nextChangeableIndex]
        tempImage = image.copy()
        tempImage[
            startCoords[0]:startCoords[0] + changeableImage.shape[0],
            startCoords[1]:startCoords[1] + changeableImage.shape[1]
            ] = changeableImage
        cv2.imshow('image', tempImage)
        params['nextChangeableIndex'] = (nextChangeableIndex + 1) % len(changeableImages)

def _createImgNameFromDict(dict):
    return '; '.join(f'{key}: {value}' for key, value in dict.items() if value is not None)

def _darkenTheseWhitePixels(image: np.array, by: int = 60):
        return cv2.threshold(image, 30, 255-by, cv2.THRESH_BINARY)[1]

@checkParameters(validParameters)
def showInLoop(images: list, 
               delay: int = 0, 
               nameParams: dict = None, 
               graphFunctions = None,
               hiddenGraphFunction = None):
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
    # show after clicking on image
    ```
    '''
    if type(graphFunctions) is not list and graphFunctions is not None:
        graphFunctions = [graphFunctions]
    if nameParams is None:
        nameParams = {}
    nImages = len(images)
    while True:
        index = 0
        while index < nImages:
            image = images[index]
            nameParams['index'] = index
            pressedKey = show(image,
                              destroy = False,
                              nameParams = nameParams,
                              delay = delay,
                              graphFunctions = graphFunctions,
                              hiddenGraphFunction = hiddenGraphFunction)
            if pressedKey == PKey:
                index = (index-1)%nImages
                continue
            if pressedKey == escKey:
                cv2.destroyAllWindows()
                return
            index += 1
            
            
def _addGraphs(graphFunctions: list, image: np.array) -> np.array:
    for graphFunction in graphFunctions:
        image = graphFunction(image)
    return image
            
def _addHiddenGraphs(hiddenGraphFunction: list, image: np.array, darkenWhitePixelsBy: int) -> list:
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
    thresholded = _darkenTheseWhitePixels(kindOfSkeleton, darkenWhitePixelsBy)
    return [(kindOfSkeleton, (0,0)), (thresholded, (0,0))]
