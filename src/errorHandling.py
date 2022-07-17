from typing import Any
import inspect
from joblib import dump

class _WrongType(Exception):
    '''
    Exception raised to indicate that there was wrong argument type\n
    found in `Argument.checkValidity` method
    '''

class _WrongValue(Exception):
    '''
    Exception raised to indicate that there was value out of valid values list\n
    found in `Argument.checkValidity` method
    '''

# TODO: obsługa `name` i `argIndex` jako listy wartości 

class Argument:
    '''
    Class that is needed to specify valid parameters for \n
    `errorHandler` and `checkArguments` decorators
    
    # Parameters:
    
    All parameters are optional, but at least two of them \n
    are needed to create instance
    
    - name: str
        - argument name
    - argType: type
    - validValues: list
    - argIndex: int
        - argument index in args and kwargs
    ---
    # Note:
    
    To use with class method specify `validParameters` key as function `__qualname__`,\n
    typically: `'<class name>.<function name>'`
    
    If used with class method, first argument (`self`) is ommited\n
    and never checked. Example:
    ```python
    validValues = {
        'Example.method' : [Argument(argType=int, argIndex=0)]
    }
    
    class Example:
        @checkParameters(validValues)
        def method(self, firstParam: int, secondParam: str):
            pass
            
    # `argIndex=0` concerns `'firstParam'` parameter
    ```
    Due to no easy way to determine whether function is class method or not\n
    during class definition (bounding with class occurs later),\n
    differentiation between common function and bounded one is based on checking\n
    first argument name - only `'self'` is treated as bounded function\n
    It does not matter when Arugments were defined without `argIndex` parameter
    '''
    def __init__(self,
                 name: str = None,
                 argType: type = None,
                 validValues: list = None,
                 argIndex: int = None):
        self._checkParameters(name, argType, validValues, argIndex)
        self._name = name
        self._argType = argType
        self._validValues = validValues
        self._argIndex = argIndex
        
    def _checkParameters(self, name, argType, validValues, argIndex):            
        if sum([item is not None for item in [name, argType, validValues, argIndex]]) < 2:
            raise AttributeError('Specify at least 2 parameters')
    
    @property
    def name(self):
        return self._name
    
    @property
    def argType(self):
        return self._argType
    
    @property
    def validValues(self):
        return self._validValues
    
    @property
    def argIndex(self):
        return self._argIndex
    
    def checkValidity(self, userValue: Any):
        if self._argType is not None:
            self._checkArgType(userValue)
        if self.validValues is not None:
            self._checkValidValues(userValue)
                
    def _checkArgType(self, userValue):
        if self._argType is not type(userValue):
            raise _WrongType
    
    def _checkValidValues(self, userValue):
        if userValue not in self._validValues:
            raise _WrongValue
        
    def __repr__(self) -> str:
        retString = 'Argument('
        retString += f'name = {self._name}, ' if self._name is not None else ''
        retString += f'argType = {self._argType.__name__}' if self._argType is not None else ''
        retString += f'validValues = {self._validValues}' if self._validValues is not None else ''
        retString += f', argIndex = {self._argIndex}' if self._argIndex is not None else ''
        retString += ')'
        return retString
        
        
class _Checker:
    def __init__(self,
                 functionName: str,
                 parametersNames: list,
                 validParameters: dict,
                 args: tuple,
                 kwargs: dict,
                 decoratedFunction: str = None):
        self.functionName = functionName
        self.decoratedFun = decoratedFunction
        self.parametersNames = parametersNames
        self.validParameters = validParameters
        self.args = args
        self.kwargs = kwargs
    
    def checkParameters(self):
        self.args = self.args if self.args[0] != 'self' else self.args[1:]
        nameInfoDict, retString = self._prepareArguments()
        for argName, (userArgVal, *argObjList) in nameInfoDict.items():
            for argObj in argObjList:
                try:
                    argObj.checkValidity(userArgVal)
                except _WrongType:
                    retString.append(f'''- {argName}: {userArgVal.join("''") if isinstance(userArgVal, str) else userArgVal} is not {argObj.argType.__name__}''')  
                except _WrongValue:
                    retString.append(f"""- {argName}: {userArgVal.join("''") if isinstance(userArgVal, str) else userArgVal} not in {argObj.validValues}""")
        funName = self.functionName if self.decoratedFun is None else self.decoratedFun.__name__
        return f'{funName}({", ".join(self.parametersNames)}):\n' + '\n'.join(retString) if retString else ''

    def _prepareArguments(self):
        retString = []
        usedParametersNames = [name for name, _ in zip(self.parametersNames, self.args)]\
                            + [name for name in self.kwargs.keys() if name in self.parametersNames]
        userArgValues = self.args+tuple(self.kwargs.values())
        nameInfoDict = {name : [value] for name, value in zip(usedParametersNames, userArgValues)}
        
        argumentsByName = {item.name : item for item in self.validParameters if item.name is not None}
        argumentsByIndex = {item.argIndex : item for item in self.validParameters if item.argIndex is not None}
        
        for name, argObj in argumentsByName.items():
            nameInfoDict[name].append(argObj)
        for index, argObj in argumentsByIndex.items():
            if len(usedParametersNames) <= index:
                neededType = f' Needed type: {argObj.argType.__name__}.' if argObj.argType is not None else ''
                neededValues = f' Needed values: {argObj.validValues}.' if argObj.validValues is not None else ''
                retString.append(f'- Argument on index {argObj.argIndex} not specified.' + neededType + neededValues)
                continue
            name = usedParametersNames[index]
            nameInfoDict[name].append(argObj)
            
        return nameInfoDict, retString
    
    
class WrongParameters(Exception):
    '''
    Exception raised when wrong parameters were passed to function \n
    in `src.fileNames` module
    '''
    def __init__(self,
                 functionName: str = None,
                 parametersNames: list = None,
                 validParameters: dict = None,
                 args: tuple = None,
                 kwargs: dict = None,
                 readyErrorMessage: str = None,
                 decoratedFunction: str = None):
        if readyErrorMessage is not None:
            super().__init__(readyErrorMessage)
            
        # has to be here so that `errorHandler` can pass it from raw exception 
        # to the one with arguments
        self.decoratedFunction = decoratedFunction
        
        if functionName is None:
            return
        self.checker = _Checker(functionName,
                                parametersNames,
                                validParameters,
                                args,
                                kwargs,
                                decoratedFunction)
        super().__init__(self.checker.checkParameters())


def errorHandler(validParameters):
    '''
    Helper decorator to make raised by function `WrongParameters` \n
    exception more precise
    '''
    def errorHandlerInner(function):
        def modFun(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except WrongParameters as e:
                decoratedFun = e.decoratedFunction
            
            parametersNames = [*inspect.signature(function if decoratedFun is None else decoratedFun).parameters]
            # In case of class methods `.__qualname__` gives '<class name>.<function name>',
            # but gives also for example: 'parallel.<locals>.processInParallel' for decorator
            # `.__name__` gives only '<function name>'
            funName = function.__qualname__ if parametersNames[0] == 'self' else function.__name__
            raise WrongParameters(funName,
                                  parametersNames, 
                                  validParameters[funName], 
                                  args, 
                                  kwargs,
                                  decoratedFunction=decoratedFun)
        return modFun
    return errorHandlerInner


def checkParameters(validParameters):
    '''
    Checks parameters before running decorated function and raise\n
    `WrongParameters` exception if any invalid parameters found\n
    Does not require raising `WrongParameters` exception in function
    '''
    def checkParametersInner(function):
        def modFun(*args, **kwargs):
            checker = _Checker(function.__name__,
                               [*inspect.signature(function).parameters],
                               validParameters[function.__name__],
                               args,
                               kwargs)
            errorMessage = checker.checkParameters()
            if errorMessage:
                raise WrongParameters(readyErrorMessage=errorMessage)
            return function(*args, **kwargs)
        return modFun
    return checkParametersInner