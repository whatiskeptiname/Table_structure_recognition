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
# Processing
Due to amount of data greater than my 8GB RAM, `src/batchProcessing.py` file comes with `parallel` decorator which enables:
- batch, parallel processing
- auto dumping results to disk
- behaviour parameterization
- input parameters validation <br>

All this without need to change implementation of processing function (except of adding one parameter that specifies decorator behavior). <br>
Example:


```python
def process1(itemsToProcess):
    # processing...
    return processedItems

# ---------------------------------------- #

@parallel
def process2(itemsToProcess, params):
    # same processing as above...
    return processedItems
```
