import numpy as np
import re

class HTMLTable:
    '''
    Class to make final modifications to created\n
    HTML code in `pd.DataFrame.to_html`.\n
    Assumes following html code format:
    
    ```html
    <table>                     <!-- starts with <table> -->
      <thead>                   <!-- optional, ignored anyway -->
        <tr>                    <!-- one tr tag per line -->
          <td>content</td>      <!-- td tag fits in one line -->
          ...
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>content</td>
          ...
        </tr>
      ...
    
    ```
    
    Ignores `thead`, `tbody` and `th` tags
    '''
    def __init__(self, structure):
        self.structure = structure
        self._code = self._getHTML().split('\n')
        self._tableTag = self._code[0]
        self._grid = self._initGrid()
        
    def _getHTML(self) -> str:
        return self.structure.generateDataFrame().to_html()
        
    def _initGrid(self) -> np.array:
        retList = []
        for line in self._code:
            line = line.strip()
            tag = re.findall('<[^>]*>', line)[0][1:-1]
            splittedTag = tag.split()
            tagName = splittedTag[0]
            tagParams = splittedTag[1:]
            if len(tagParams):
                tagParams[-1] = tagParams[-1][:-1]
            if tagName == 'tr':
                currentRow = []
            elif tagName == '/tr':
                if currentRow:
                    retList.append(currentRow)
            elif tagName == 'td':
                content = re.findall('>.*<', line)[0][1:-1]
                currentRow.append(HTMLCell(content))
        return np.array(retList, dtype=object)
    
    def _getColSpanAttrib(self, nCols) -> str:
        return f'colspan="{nCols}"'
    
    def _getRowSpanAttrib(self, nRows) -> str:
        return f'rowspan="{nRows}"'
    
    def _spanColumns(self) -> None:
        for coords in self.structure.gridCoords:
            rowLeft, rowRight = coords.row
            colTop, colDown = coords.col
            
            nSpanRows = rowRight - rowLeft + 1
            nSpanCols = colDown - colTop + 1
            if nSpanRows > 1:
                self._grid[colTop, rowLeft].attribs.append(self._getColSpanAttrib(nSpanRows))
                for cell in self._grid[colTop, rowLeft+1 : rowRight+1]:
                    cell.toDelete = True
            if nSpanCols > 1:
                self._grid[colTop, rowLeft].attribs.append(self._getRowSpanAttrib(nSpanCols))
                for cell in self._grid[colTop+1 : colDown+1, rowLeft]:
                    cell.toDelete = True
                
        self._grid = [[cell for cell in row if not cell.toDelete] for row in self._grid]
        
    def getFormattedCode(self) -> str:
        '''
        Method that returns processed HTML table code
        '''
        formattedCode = [self._tableTag]
        for row in self._grid:
            formattedCode.append('<tr>')
            for cell in row:
                formattedCode.append(
                    f"<td {' '.join(cell.attribs)}>{cell.content}</td>"
                    )
            formattedCode.append('</tr>')
        formattedCode.append('</table>')
        return ' '.join(formattedCode)
                    

class HTMLCell:
    # td
    def __init__(self, content: str, attribs: list = None):
        self.content = content
        self.attribs = [] if attribs is None else attribs
        self.toDelete = False
    
    def __repr__(self):
        return f"'{self.content}'"        
