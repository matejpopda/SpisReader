from typing import Optional

import pyvista.core.composite


class DataSetFilters:
        ...
        def slice_orthogonal(self, x:Optional[float]=None, y:Optional[float]=None, z:Optional[float]=None, generate_triangles:bool=False, contour:bool=False, progress_bar:bool=False) -> pyvista.core.composite.MultiBlock:
                ...
    