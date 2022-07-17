from joblib import Parallel, delayed, cpu_count, dump, load
import os
import warnings
import traceback
import time
import numpy as np
from tqdm import tqdm

import src.fileNames as fileNames
from src.errorHandling import Argument, errorHandler, WrongParameters


validParameters = {
    'processInParallel' : [Argument(argType=list, argIndex=0),
                           Argument(argType=dict, argIndex=1)]
}

class TqdmParallel(Parallel):
    '''
    Overwritten `joblib.Parallel` class to enable showing progress using `tqdm`
    
    # Parameters:
    ---
    nParts: int
        Amount of parts data was splitted on; passed as `total` parameter to `tqdm`
    n_jobs: int
        First argument passed to `Parallel`\n
        totally no need to be here, but I was passing it either way and wanted\n
        it to be here so I know why I passed two parameters
    '''
    def __init__(self, nParts: int, n_jobs: int, *args, **kwargs):
        self.nParts = nParts
        super().__init__(n_jobs, *args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        with tqdm(total=self.nParts) as self._tqdm:
            return Parallel.__call__(self, *args, **kwargs)
        
    def print_progress(self):
        self._tqdm.n = self.n_completed_tasks
        self._tqdm.refresh()
        
        
class _ParallelHelper:
    '''
    Helper class that stores parameters and methods used in `src.batchProcessing.parallel`.\n
    Used and written only for `parallel`
    '''
    def __init__(self, params, decoratedFunction):
        self.startTime = time.time()
        self._params = params
        self.testType = params.get('testType')
        self.trialIndex = params.get('trialIndex')
        self.returnResults = params.get('returnResults', False)
        self.storeResults = params.get('storeResults', True)
        self.compressMethod = params.get('compressMethod', None)
        self.compressLevel = params.get('compressLevel', 3)
        self.nParts = params.get('nParts', 8)
        self.decoratedFunction = decoratedFunction
    
    @property
    def params(self):
        self._params['totalSize'] = self._getTotalSize()
        self._params['totalTime'] = round(time.time() - self.startTime, 3)
        return self._params

    def splitData(self, data):
        # TODO: better data splitting
        nRecordsInSplit = len(data)/(self.nParts-1)
        return [data[index:index+int(nRecordsInSplit)] for index in range(0, len(data), int(nRecordsInSplit))]
    
    def process(self, dumpfileName):
        '''
        Main method of `_ParallelHelper` that is responsible for running decorated function
        '''
        def modFun(*args, **kwargs):
            # dumpfileName = args[-1]
            # args = args[:-1]
            try:
                results = self.decoratedFunction(*args, **kwargs)
            except:
                self.dumpData(traceback.format_exc(), f'error{dumpfileName}')
                return dumpfileName, False, None
            self.dumpData(results, dumpfileName)
            if self.returnResults:
                return dumpfileName, True, results
            return dumpfileName, True, None
        return modFun
    
    def dumpData(self, data, fileName):
        if not self.storeResults:
            return
        storeLocation = self.getTrialFolder(fileName)
        dump(data, storeLocation, compress=self.compressMethod)
        
    def getTrialFolder(self, fileName = ''):
        try:
            return fileNames.getTestsTrialDir(self.testType, self.trialIndex, fileName)
        except:
            raise WrongParameters(decoratedFunction=self.decoratedFunction)
        
    
    def getSuccessfulResults(self, parallelReturn):
        # parallelReturn[index][0] -> fileName
        # parallelReturn[index][1] -> returnStatus
        # parallelReturn[index][2] -> results
        if not all(item[1] for item in parallelReturn):
            failedList = [item[0] for item in parallelReturn if not item[1]]
            warnings.warn(f'{len(failedList)}/{len(parallelReturn)} processes failed.\
                        \nFailed saving data to files: {failedList}')
            
        if not self.returnResults:
            return None
        successfulResults = [item[2] for item in parallelReturn if item[1] and item[2] is not None]
        self._params['processedItems'] = len(successfulResults)
        try:
            if len(successfulResults[0]) > 1:
                # `function` returns more than one value
               successfulResults = self._reorderResults(successfulResults)
        except TypeError:
            # `function` does not return list -> returned object has no len()
            pass
        return successfulResults
        
    def _getTotalSize(self):
        totalSize = 0
        for file in os.scandir(self.getTrialFolder()):
            if file.name != 'metadata':
                totalSize += os.path.getsize(file)
        return totalSize
    
    def _reorderResults(self, results):
        transposed = np.array(results, dtype=object).transpose()
        unnestedResults = []
        for index, item in enumerate(transposed):
            try:
                unnestedResults.append(np.concatenate(item))
            except:
                unnestedResults.append(item)
        return tuple(unnestedResults)


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
        - `'compressMethod'`: `str` [optional], default: `None`
            - parameter `compress` in `joblib.dump`
        - `'compressLevel'`: `int` [optional], default: `3`
            - parameter `compress` in `joblib.dump`
        - `'nParts'`: int [optional], default: `8`
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
        try:
            data = args[0]
            params = args[1]
            info = _ParallelHelper(params, function)
            data = info.splitData(data)
        except (IndexError, AttributeError, TypeError):
            # `IndexError` when there are not enough args
            # `AttributeError` when `params` is not dict
            # `TypeError` when `data` has no len()
            raise WrongParameters(decoratedFunction=function)
        
        parallelReturn = TqdmParallel(info.nParts, cpu_count())(
                                        delayed(info.process(str(fileIndex)))
                                        (toProcess, *args[1:], **kwargs) for fileIndex, toProcess in enumerate(data))
        
        successfulResults = info.getSuccessfulResults(parallelReturn)
        info.dumpData(info.params, 'metadata')
        return successfulResults
        
    return processInParallel


def loadTrial(testType, trialIndex, firstN = None) -> tuple:
    '''
    Loads data stored by `parallel`.
    
    # Returns:\n
    `tuple`: (data, metadata) if 'metadata' file was found in trial folder\n
    `tuple`: (data, `None`) if 'metadata' file was not found in trial folder
    '''
    data = loadTrialData(testType, trialIndex)
    metadata = loadTrialMetadata(testType, trialIndex)
    return data, metadata

def loadTrialData(testType, trialIndex, firstN = None):
    trialPath = fileNames.getTestsTrialDir(testType, trialIndex)
    filesToLoad = [file for file in os.listdir(trialPath)
                   if file != 'metadata'
                   and not file.startswith('error')]
    filesToLoad = filesToLoad[:firstN]
    data = [y for x in [load(f'{trialPath}/{fileToLoad}') for fileToLoad in filesToLoad] for y in x]
    return data
    

def loadTrialMetadata(testType, trialIndex):
    try:
        filePath = fileNames.getTestsTrialDir(testType, trialIndex, 'metadata')
        return load(filePath)
    except FileNotFoundError:
        return None

def loadErrors(testType, trialIndex):
    trialPath = fileNames.getTestsTrialDir(testType, trialIndex)
    filesToLoad = [file for file in os.listdir(trialPath) if file.startswith('error')]
    return [load(f'{trialPath}/{fileToLoad}') for fileToLoad in filesToLoad]


def getParamsDict(testType: str = None,
                  trialIndex: int = None,
                  storeResults: bool = True,
                  returnResults: bool = False,
                  compressMethod = None,
                  compressLevel = 0,
                  nParts = 8) -> dict:
    '''
    Returns simple parameters dictionary needed in function\n
    decorated by `parallel`
    '''
    return {'testType': testType,
            'trialIndex': trialIndex,
            'storeResults': storeResults,
            'returnResults': returnResults,
            'compressMethod': compressMethod,
            'compressLevel': compressLevel,
            'nParts': nParts}