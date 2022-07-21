from joblib import Parallel, delayed, cpu_count, dump, load
import os
import warnings
import traceback
import time
import datetime
import numpy as np
from tqdm import tqdm

import src.fileNames as fileNames
from src.errorHandling import Argument, errorHandler, WrongParameters
from src.storage import Storage


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
        self._params = params.copy()
        self.testType = params.get('testType')
        self.trialIndex = params.get('trialIndex')
        self.returnResults = params.get('returnResults', False)
        self.storeResults = params.get('storeResults', True)
        self.compressMethod = params.get('compressMethod', 'zlib')
        self.compressLevel = params.get('compressLevel', 3)
        self.nParts = params.get('nParts', 8)
        self.decoratedFunction = decoratedFunction
        
        self._dumpCompression = self._getDumpCompression()
        self._params['parallelData'] = {}
    
    @property
    def params(self):
        self._addMetadata('functionName', self.decoratedFunction.__name__)
        self._addMetadata('totalSize', self._getTotalSize())
        self._addMetadata('date', self._getDate())
        self._addMetadata('totalTime', round(time.time() - self.startTime, 3))
        return self._params
    
    def _addMetadata(self, paramName, value):
        self._params['parallelData'][paramName] = value
        
    def _getDate(self):
        return datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S.%f')

    def splitData(self, data):
        # TODO: better data splitting
        nRecordsInSplit = len(data)/(self.nParts-1)
        return [data[index:index+int(nRecordsInSplit)] for index in range(0, len(data), int(nRecordsInSplit))]
    
    def _getDumpCompression(self):
        if self.compressMethod and self.compressLevel:
            return (self.compressMethod, self.compressLevel)
        return self.compressMethod or self.compressLevel
    
    def process(self, dumpfileName):
        '''
        Main method of `_ParallelHelper` that is responsible for running decorated function
        
        Returns `tuple` with:
            - dump file name: `str`
            - success status: `bool`
                - `True` when success
                - `False` when any error
            - results from decorated function:
                - `list` if `params['returnResults'] == True`
                - `None` if `params['returnResults'] == False`
        '''
        def modFun(*args, **kwargs):
            try:
                results = self.decoratedFunction(*args, **kwargs)
            except:
                error = {
                    'testType' : self.testType,
                    'trialIndex' : self.trialIndex,
                    'functionName' : self.decoratedFunction.__name__,
                    'date' : self._getDate(),
                    'traceback': traceback.format_exc()
                }
                Storage.saveError(error)
                return dumpfileName, False, None
            
            if self.storeResults:
                Storage.saveData(results, 
                                 self._getTrialFolder(dumpfileName),
                                 self._dumpCompression)
            if self.returnResults:
                return dumpfileName, True, results
            return dumpfileName, True, None
        return modFun
        
    def _getTrialFolder(self, fileName = ''):
        try:
            return fileNames.getTestsTrialDir(self.testType, self.trialIndex, fileName)
        except:
            raise WrongParameters(decoratedFunction=self.decoratedFunction)
        
    
    def getSuccessfulResults(self, parallelReturn, nItemsInParts):
        # Based on `self.process`:
        # parallelReturn[index][0] -> fileName
        # parallelReturn[index][1] -> returnStatus
        # parallelReturn[index][2] -> results
        
        dumpNames, returnStatuses, results = zip(*parallelReturn)
        
        # errors checking
        nFailedItems = 0
        if not all(returnStatuses):
            failedList, nFailedItems = zip(*[(fileName, nItems) 
                                             for fileName, returnStatus, nItems 
                                             in zip(dumpNames, returnStatuses, nItemsInParts) 
                                             if not returnStatus])
            nFailedItems = sum(nFailedItems)    
            warnings.warn(f'{len(failedList)}/{len(parallelReturn)} processes failed.\
                        \nFailed saving data to files: {failedList}')
 
        # saving metadata
        self._addMetadata('nProcessedItems', sum(nItemsInParts) - nFailedItems)
        self._addMetadata('nItems', sum(nItemsInParts))
        
        # getting only successfuly processed results
        if not self.returnResults:
            return None
        successfulResults = [result
                             for result, returnStatus 
                             in zip(results, returnStatuses) 
                             if returnStatus]
        
        # transposing and concatenating results if more than value if returned
        # by processing function
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
        for file in os.scandir(self._getTrialFolder()):
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
        - `'compressMethod'`: `str` [optional], default: `zlib`
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

        nItemsInParts = [len(part) for part in data]
        successfulResults = info.getSuccessfulResults(parallelReturn, nItemsInParts)
        Storage.saveMetadata(info.params)
        return successfulResults
        
    return processInParallel

def getParamsDict(testType: str = None,
                  trialIndex: int = None,
                  storeResults: bool = True,
                  returnResults: bool = False,
                  compressMethod = 'zlib',
                  compressLevel = 3,
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