
from pathlib import Path
import pyvista.plotting
import spisModule.reader as reader
import spisModule.simulation as simulation
import pyvista.core.dataset
import logging 
log = logging.getLogger(__name__)


# For Mesh
    # Slice - takes origin and normal, and mesh
    # Slice origin - just normal
    # Define constants that are the directions of normals
    # Plot the slice
    # Glob for properties - order them by time (final at the end) 
    # Draw the whole list in a gif
    # Draw the list into separate files
    # Get a list of properties somewhere
    # 

# For timeseries 
    # Draw it

# Better access for some stuff in the simulation through @property


class Directions:
    x_plus  = (1,0,0)
    x_minus = (-1, 0, 0)
    y_plus  = (0,1,0)
    y_minus = (0,-1,0)
    z_plus  = (0,0,1)
    z_minus = (0,0,-1)

class PlaneNormals:
    XY = (0,0,1)
    XZ = (0,1,0)
    YZ = (1,0,0)
    XY_flipped = (0,0,-1)
    XZ_flipped = (0,-1,0)
    YZ_flipped = (-1,0,0)



dataset = pyvista.core.dataset.DataSet|simulation.Mesh
vector = Directions|PlaneNormals|tuple[float, float, float]|tuple[int, int, int]



def interactive_plot_orth_slice(mesh: dataset, property:str) -> None:
    interactive_plot_mesh(mesh.slice_orthogonal(), property=property) # type: ignore

def interactive_plot_physical_mesh(mesh: dataset) -> None:
    interactive_plot_mesh(mesh, "gmsh:physical")


def interactive_plot_mesh(mesh: dataset, property:str) -> None:
    if isinstance(mesh, reader.Mesh):
        mesh = mesh.mesh

    plotter = pyvista.plotting.Plotter() # type: ignore
    plotter.add_mesh(mesh, scalars=property)  # type: ignore
    plotter.show()     # type: ignore


def save_mesh(mesh: dataset, property:str, path:Path = Path("./temp"), filename:str|None=None) -> None:
    if isinstance(mesh, reader.Mesh):
        mesh = mesh.mesh

    if not path.exists(): 
        log.warn(f"Output folder {str(path.resolve())} does not exist, creating it")
        path.mkdir()   

    if filename is None:
        filename = property + ".png"

    path = path/filename

    plotter = pyvista.plotting.Plotter(off_screen=True) # type: ignore
    plotter.add_mesh(mesh, scalars=property)  # type: ignore
    plotter.screenshot(filename=path,)     # type: ignore


def slice_and_save(mesh: dataset, property:str, normal: vector, *, slice_origin:vector=(0,0,0) ,path:Path = Path("./temp"), filename:str|None=None) -> None:
    if isinstance(mesh, reader.Mesh):
        mesh = mesh.mesh

    if not path.exists(): 
        log.warn(f"Output folder {str(path.resolve())} does not exist, creating it")
        path.mkdir()   

    if filename is None:
        filename = property + ".png"

    path = path/filename

    mesh=mesh.slice(normal=normal, origin=slice_origin) # type: ignore

    plotter = pyvista.plotting.Plotter(off_screen=True) # type: ignore
    plotter.add_mesh(mesh, scalars=property)  # type: ignore

    plotter.enable_parallel_projection()  # type: ignore
    plotter.camera_position = normal

    plotter.screenshot(filename=path, scale=10)     # type: ignore