
from pathlib import Path
import pyvista.plotting
from pyvista.plotting.plotter import Plotter
from pyvista.core.dataset import DataSet
import spisModule.simulation as simulation
import pyvista.core.dataset
from spisModule.helpers import allow_mesh, check_and_create_folder
import logging 
log = logging.getLogger(__name__)


SCREENSHOT_SIZE = 2 

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


vector = Directions|PlaneNormals|tuple[float, float, float]|tuple[int, int, int]




@allow_mesh
def interactive_plot_orth_slice(mesh: DataSet, property:str) -> None:
    temp: DataSet = mesh.slice_orthogonal()
    interactive_plot_mesh(temp, property=property)  


@allow_mesh
def interactive_plot_physical_mesh(mesh: DataSet) -> None:
    interactive_plot_mesh(mesh, "gmsh:physical")

@allow_mesh
def interactive_plot_mesh(mesh: DataSet, property:str) -> None:

    plotter = Plotter()  
    plotter.add_mesh(mesh, scalars=property)   
    plotter.show()      


@allow_mesh
def save_mesh(mesh: DataSet, 
              property:str, 
              path:Path = Path("./temp"), 
              filename:str|None=None,
              *,
              screenshot_size:int = SCREENSHOT_SIZE,) -> None:
    check_and_create_folder(path)
    filename = _default_filename(filename=filename, property=property)


    path = path/filename

    plotter = Plotter(off_screen=True)  
    plotter.add_mesh(mesh, scalars=property)   
    plotter.screenshot(filename=path,scale=screenshot_size)      


@allow_mesh
def slice_and_save(mesh: DataSet, 
                   property:str, 
                   normal: vector, 
                   *, 
                   slice_origin:vector=(0,0,0),
                   path:Path = Path("./temp"), 
                   filename:str|None=None,
                   screenshot_size:int = SCREENSHOT_SIZE,
                   ) -> None:
    
    check_and_create_folder(path)
    filename = _default_filename(filename=filename, property=property)


    path = path/filename

    mesh=mesh.slice(normal=normal, origin=slice_origin)  

    plotter = pyvista.plotting.Plotter(off_screen=True)  
    plotter.add_mesh(mesh, scalars=property)   

    plotter.enable_parallel_projection()   
    plotter.camera_position = normal

    plotter.screenshot(filename=path, scale=screenshot_size)      


def _default_filename(filename:str|None, property:str) -> str:
    if filename is None:
        filename = property + ".png"
    return filename