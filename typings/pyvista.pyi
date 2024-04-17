import meshio._mesh as mesh
from pyvista.core.pointset import UnstructuredGrid
from typing import overload

@overload
def wrap(dataset: mesh.Mesh) -> UnstructuredGrid: ...
@overload
def wrap(dataset: None) -> None: ...
