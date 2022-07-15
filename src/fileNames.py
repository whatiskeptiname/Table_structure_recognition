import os
from src.errorHandling import errorHandler, WrongParameters, Argument
from pathlib import Path

testsDir = './tests'
testTypes = ['compression_time']

validParameters = {
    'getImages' : [Argument('set_', validValues=['train', 'test'])],
    'getStructure' : [Argument('set_', validValues=['train', 'test'])],
    'getTestsDir' : [Argument('testType', validValues=testTypes)],
    'getTestsTrial' : [Argument('testType', validValues=testTypes),
                       Argument('trialIndex', int),
                       Argument('fileName', str)]
}


dataDir = './data'
trainDir = f'{dataDir}/train'
testDir = f'{dataDir}/test'
trainImagesDir = f'{trainDir}/img'
testImagesDir = f'{testDir}/img'
trainStructureDir = f'{trainDir}/structure'
testStructureDir = f'{testDir}/structure'
deletedDataDir = f'{dataDir}/deleted'
deletedTrainDataDir = f'{deletedDataDir}/train'
deletedTestDataDir = f'{deletedDataDir}/test'


@errorHandler(validParameters)
def getImages(set_: str = 'train'):
    if set_ == 'train':
        return [f'{trainImagesDir}/{imageName}' for imageName in os.listdir(trainImagesDir)]
    if set_ == 'test':
        return [f'{testImagesDir}/{imageName}' for imageName in os.listdir(testImagesDir)]
    raise WrongParameters

@errorHandler(validParameters)
def getStructure(set_: str = 'train'):
    if set_ == 'train':
        return [f'{trainStructureDir}/{imageName}' for imageName in os.listdir(trainStructureDir)]
    if set_ == 'test':
        return [f'{testStructureDir}/{imageName}' for imageName in os.listdir(testStructureDir)]
    raise WrongParameters

@errorHandler(validParameters)
def getTestsDir(testType: str):
    testTypeDir = f'{testsDir}/{testType}'
    Path(testTypeDir).mkdir(parents=True, exist_ok=True)
    return testTypeDir

@errorHandler(validParameters)
def getTestsTrialDir(testType: str, trialIndex: int, fileName: str = ''):
    trialDir = f'{getTestsDir(testType)}/{trialIndex}'
    Path(trialDir).mkdir(parents=True, exist_ok=True)
    return f'{trialDir}/{fileName}' if fileName else trialDir

def moveToDeleted(filePath):
    newFilePath = filePath.replace('data', 'data/deleted').replace('/img', '')
    Path(deletedTrainDataDir).mkdir(parents=True, exist_ok=True)
    Path(deletedTestDataDir).mkdir(parents=True, exist_ok=True)
    os.rename(filePath, newFilePath)
    