from typing import Optional


import pyvista
from pyvista.core.utilities.misc import abstract_class
import pyvista.core.composite



@abstract_class
class DataSetFilters:
        ...
        def slice_orthogonal(self, 
                             x:Optional[float]=None, 
                             y:Optional[float]=None, 
                             z:Optional[float]=None, 
                             generate_triangles:bool=False, 
                             contour:bool=False, 
                             progress_bar:bool=False) -> pyvista.core.composite.MultiBlock:
                ...
