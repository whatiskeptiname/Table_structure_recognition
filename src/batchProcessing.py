from joblib import Parallel, delayed, cpu_count, dump, load
import os
import warnings
import traceback
import time

import src.fileNames as fileNames
from src.errorHandling import Argument, errorHandler, WrongParameters, dumpError


validParameters = {
    'processInParallel' : [Argument(argType=list, argIndex=0),
                           Argument(argType=dict, argIndex=1)]
}


def parallel(function):
    '''
    Decorator to make processing function parallel without changing it's implementation
    
    # Requires function to have:
    - first parameter: `list` of items to process
    - second parameter: parameters dict with keys:
        - `'testType'`: `str`
            - folder name of that kind of processing, optional if `'storeResults': False`
        - `'trialIndex'`: `int`
            - folder name in `testType`, optional if `'storeResults': False`
        - `'returnResults'`: `bool` [optional], default: `False`
        - `'storeResults'`: `bool` [optional], default: `True`
        - `'compressMethod'`: `int`/`str` [optional], default: `0`
            - parameter `compress` in `joblib.dump`
        - `'nSplits'`: int [optional], default: `8`
            - amount of splits on processed data, `nParts-1` splits are equal
    - any other needed parameters
    
    # Returns:\n
    If `'returnResults': True` returns list with succesfully processed data, \n
    otherwise returns `None`
    
    # Note:\n
    If any of processing units fails, this function \n
    shows accumulated warning and continues
    '''
    
    @errorHandler(validParameters)
    def processInParallel(*args, **kwargs):
        startTime = time.time()
        try:
            data = args[0]
            params = args[1]
            data = _splitData(data, params)
        except (IndexError, AttributeError, TypeError):
            # `IndexError` when there are not enough args
            # `AttributeError` when `params` is not dict
            # `TypeError` when `data` has no len()
            raise WrongParameters(decoratedFunction=function)
        
        parallelReturn = Parallel(cpu_count())(
                                        delayed(_saveResults(function))
                                        (toProcess, params, *args[2:], str(fileIndex), **kwargs) for fileIndex, toProcess in enumerate(data))
        # parallelReturn[index][0] -> fileName
        # parallelReturn[index][1] -> returnStatus
        # parallelReturn[index][2] -> results
        
        if not all(item[1] for item in parallelReturn):
            failedList = [item[0] for item in parallelReturn if not item[1]]
            warnings.warn(f'{len(failedList)}/{len(parallelReturn)} processes failed.\
                        \nFailed save data to files: {failedList}')
        
        totalTime = time.time() - startTime
        params['processingTime'] = round(totalTime, 3)
        _dumpData(params, params, 'metadata')
        successfulResults = [item[2] for item in parallelReturn if item[1] and item[2] is not None]
        
        if params.get('returnResults', False) and successfulResults:
            return [y for x in successfulResults for y in x]
        return None
    return processInParallel

def _splitData(data, params):
    nSplits = params.get('nSplits', 8)
    nRecordsInSplit = len(data)/nSplits
    return [data[index:index+int(nRecordsInSplit)] for index in range(0, len(data), int(nRecordsInSplit))]

def _dumpData(data, params, fileName, function):
    storeResults = params.get('storeResults', True)
    if not storeResults:
        return
    testType = params.get('testType')
    trialIndex = params.get('trialIndex')
    if testType is None or trialIndex is None:
        raise WrongParameters(decoratedFunction=function)
    compressMethod = params.get('compressMethod', 0)
    storeLocation = fileNames.getTestsTrialDir(testType, trialIndex, fileName)
    dump(data, storeLocation, compress=compressMethod)
    
def _saveResults(function):
    def modFun(*args, **kwargs):
        dumpfileName = args[-1]
        args = args[:-1]
        params = args[1]
        returnResults = params.get('returnResults', False)
        try:
            results = function(*args, **kwargs)
        except:
            errorPath = fileNames.getTestsTrialDir(params.get('testType'), params.get('trialIndex'), f'error{dumpfileName}')
            errorMessage = traceback.format_exc()
            dumpError(errorPath, errorMessage)
            return dumpfileName, False, None
        _dumpData(results, params, dumpfileName, function)
        if returnResults:
            return dumpfileName, True, results
        return dumpfileName, True, None
    return modFun


def loadTrial(testType, trialIndex, firstN = None) -> tuple:
    '''
    Loads data stored by `parallel`.
    
    # Returns:\n
    `tuple`: (data, metadata) if 'metadata' file was found in trial folder\n
    `tuple`: (data, None) if 'metadata' file was not found in trial folder
    '''
    trialPath = fileNames.getTestsTrialDir(testType, trialIndex)
    filesToLoad = os.listdir(trialPath)
    filesToLoad = filesToLoad if firstN is None else filesToLoad[:firstN]
    data = [load(f'{trialPath}/{fileToLoad}') for fileToLoad in filesToLoad][0]
    metadata = None
    if os.path.exists(f'{trialPath}/metadata'):
        metadata = load(f'{trialPath}/metadata')
    return data, metadata


def getParamsDict(testType: str = None,
                  trialIndex: int = None,
                  storeResults: bool = True,
                  returnResults: bool = False,
                  compressMethod = 0,
                  nSplits = 8) -> dict:
    '''
    Returns simple parameters dictionary needed in function\n
    decorated by `parallel`
    '''
    return {'testType': testType,
            'trialIndex': trialIndex,
            'storeResults': storeResults,
            'returnResults': returnResults,
            'compressMethod': compressMethod,
            'nSplits': nSplits}