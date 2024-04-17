import pathlib
import meshio._mesh as mesh

def read(path: pathlib.Path, file_format: str | None = None) -> mesh.Mesh: ...
