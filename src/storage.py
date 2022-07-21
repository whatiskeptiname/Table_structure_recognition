import pymongo
import os
from joblib import load, dump
import src.fileNames as fileNames


class _Database:
    _dbName = 'imageProcessing'
    _client = pymongo.MongoClient('localhost', 27017)
    _db = _client[_dbName]
    _metadataCollection = _db['metadata']
    _errorCollection = _db['errors']
    
    @classmethod
    def saveMetadata(self, metadata):
        self._metadataCollection.insert_one(metadata)
        
    @classmethod
    def loadMetadata(self, *args, **kwargs):
        '''
        Loads metadata from database
        '''
        cursor = self._metadataCollection.find(*args, **kwargs)
        return [*cursor]
    
    @classmethod
    def saveError(self, error):
        self._errorCollection.insert_one(error)
        
    @classmethod
    def loadError(self, *args, **kwargs):
        '''
        Loads errors from database
        '''
        cursor = self._errorCollection.find(*args, **kwargs)
        return [*cursor]
        
    
class _Files:
    @classmethod
    def loadData(self, testType, trialIndex, firstN = None) -> list:
        '''
        Loads dumped data from file system
        '''
        trialPath = fileNames.getTestsTrialDir(testType, trialIndex)
        filesToLoad = os.listdir(trialPath)[:firstN]
        data = [y for x in [load(f'{trialPath}/{fileToLoad}') for fileToLoad in filesToLoad] for y in x]
        return data
    
    @classmethod
    def saveData(self, data, storePath, compression):
        dump(data, storePath, compression)
        

    
class Storage(_Database, _Files):
    '''
    Static class that provides access to any data stored in both\n
    database and file system
    '''
    
    @classmethod
    def loadTrial(self, testType, trialIndex) -> tuple:
        '''
        Loads trial data, metadata and errors
        '''
        args = {
            'testType' : testType,
            'trialIndex' : trialIndex
        }
        return (self.loadData(testType, trialIndex),
                self.loadMetadata(args),
                self.loadError(args))
    
    
    
    



    
    

    