# Graphical table structure recognition

### Converting tabular data from .png images to HTML/LaTeX equivalent using image processing tools

*So far there are only preparations made, no image processing included

---
# Data origin:
<details>
<summary>SciTSR</summary>
<a href="https://github.com/Academic-Hammer/SciTSR">link</a>
<br>
Dataset containing 15 000 table images and their corresponding LaTeX structure
</details>
<details>
<summary>PubTabNet</summary>
<a href="https://github.com/ibm-aur-nlp/PubTabNet/tree/master/src">link</a>
<br>
Dataset containing >568 000 table images as well as corresponding HTML structure labels
</details>

---
# Parallel processing
Due to amount of data greater than my 8GB RAM, `src/batchProcessing.py` file comes with `parallel` decorator which enables:
- batch, parallel processing
- auto dumping results and metadata to disk
- behaviour parameterization
- showing <a href=https://tqdm.github.io/>`tqdm`</a> progress bar
- input parameters validation <br>

All this without need to change implementation[^1] of processing function <br>(except for first parameter that has to be `list` and the second one that has to be `dict` that specifies decorator behavior). <br>
Example:


```python
def process1(itemsToProcess):
    # linear processing...
    return processedItems, otherReturnValue

# ---------------------------------------- #

@parallel
def process2(itemsToProcess, params):
    # same processing as above...
    return processedItems, otherReturnValue

# ---------------------------------------- #

# Wrong input parameter(s) raise `WrongParameters` exception 
process2(5, 'foo')

# Error message:
WrongParameters: process2(itemsToProcess, params):
- itemsToProcess: 5 is not list
- params: 'foo' is not dict

```

`WrongParameters` works with both args and kwargs passed in any order by using `errorhandler` decorator from `src/errorhandling.py` with parameters as `List[Argument]`.<br>
`Argument` object enables specyfing function parameters types or valid values by their names or/and indexes in args/kwargs.





[^1]: Return value is the same until:<br>
\- `params['returnResults'] == False`<br>
\- function is used inside `try...except` block and exception occures -> due to multiprocessing<br>
\- function changes processed items order -> due to splitting data and processing separately<br>
\- probably in some other edge cases

