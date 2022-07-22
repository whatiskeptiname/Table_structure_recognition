import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from src.errorHandling import checkParameters, Argument
from types import FunctionType

PKey = 112
escKey = 27

validParameters = {
    'showInLoop' : [Argument('images'), list,
                    Argument('nameParams', dict),
                    Argument('graphFunctions', FunctionType)]
}


def show(image, destroy = True, nameParams = None, delay = 0, graphFunctions = None):
    if nameParams is None:
        nameParams = {}
    if np.unique(image).size == 2:
        image = 255*image
    if graphFunctions is not None:
        image = _addGraphs(graphFunctions, image)
    cv2.imshow('image', image)
    mouseCallbackParameters = {
        'image' : image,
        'nameParams' : nameParams
    }
    cv2.setMouseCallback('image', _mouseEvent, mouseCallbackParameters)
    if nameParams is not None:
        cv2.setWindowTitle('image', _createImgNameFromDict(nameParams))
    pressedKey = cv2.waitKey(delay)
    if not destroy:
        return pressedKey
    cv2.destroyAllWindows()
    
def _mouseEvent(event, x, y, flags, params):
    if event == cv2.EVENT_MOUSEMOVE:
        newImgName = params['nameParams'].copy()
        newImgName.update({
            'coords': (y,x),
            'color': params['image'][y,x]
            })
        cv2.setWindowTitle('image', _createImgNameFromDict(newImgName))

def _createImgNameFromDict(dict):
    return '; '.join(f'{key}: {value}' for key, value in dict.items() if value is not None)

@checkParameters(validParameters)
def showInLoop(images: list, delay: int = 0, nameParams: dict = None, graphFunctions = None):
    '''
    Function to enable showing many images in a loop.\n
    By default shows cursor position and related pixel color in a title field
    
    # Note:
    - press 'p' to display previous image
    - press 'esc' to close
    - press any other key to display next image
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
                              graphFunctions = graphFunctions)
            if pressedKey == PKey:
                index = (index-1)%nImages
                continue
            if pressedKey == escKey:
                cv2.destroyAllWindows()
                return
            index += 1
            
            
def _addGraphs(graphFunctions, image) -> np.array:
    for graphFunction in graphFunctions:
        image = graphFunction(image)
    return image
            
def addPlot(image) -> np.array:
    '''
    Function to add vertical and horizontal plots to image\n
    showing aggregated vertical and horizontal pixels sums
    '''
    hists = []
    for axis in [0,1]:
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.axis('off')
        ax.margins(0)
        fig.tight_layout(pad=0)

        ax.plot(image.sum(axis=axis))

        fig.canvas.draw()
        histImage = np.frombuffer(fig.canvas.tostring_rgb(), dtype = 'uint8')
        histImage = histImage.reshape(fig.canvas.get_width_height()[::-1] + (3,))[:,:,1]
        histImage = cv2.threshold(histImage, 253, 255, cv2.THRESH_BINARY)[1]
        if axis == 0:
            dsize = (image.shape[1], histImage.shape[0])
        else:
            histImage = histImage.transpose()
            dsize = (histImage.shape[1], image.shape[0])
        histImage = cv2.resize(histImage, dsize)
        hists.append(histImage)
        
    hists[1] = np.append(hists[1], 
                         np.zeros((hists[0].shape[0], hists[1].shape[1]), dtype='uint8'),
                         axis=0)
    retImage = np.append(
        image, 
        hists[0], 
        axis=0)
    retImage = np.append(
        retImage, 
        hists[1], 
        axis=1)
    return retImage

