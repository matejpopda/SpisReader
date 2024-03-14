import meshio._mesh as mesh
from pyvista.core.dataset import DataSet
from typing import overload

@overload
def wrap(dataset: mesh.Mesh) -> DataSet:
    ...

@overload
def wrap(dataset: None) -> None:
    ...

