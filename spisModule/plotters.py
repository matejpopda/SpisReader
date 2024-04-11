
from pathlib import Path
import pyvista.plotting
from pyvista.plotting.plotter import Plotter
from pyvista.core.dataset import DataSet
from spisModule.simulation import *
import pyvista.core.dataset
from spisModule.helpers import allow_mesh, check_and_create_folder
import logging 
import fnmatch
log = logging.getLogger(__name__)


SCREENSHOT_SIZE = 2 

ORIGIN_VECTOR = (0,0,0)
DEFAULT_PATH = Path("./temp")


# For Mesh
    # Slice - takes origin and normal, and mesh
    # Slice origin - just normal
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


def _default_filename(filename:str|None, property:str) -> str:
    if filename is None:
        filename = property + ".png"
    return filename


@allow_mesh
def interactive_plot_orth_slice(mesh: DataSet, property:str) -> None:
    temp: DataSet = mesh.slice_orthogonal()             #type: ignore
    interactive_plot_mesh(temp, property=property)      #type: ignore


@allow_mesh
def interactive_plot_physical_mesh(mesh: DataSet) -> None:
    interactive_plot_mesh(mesh, "gmsh:physical")

@allow_mesh
def interactive_plot_mesh(mesh: DataSet, property:str) -> None:

    plotter = Plotter()  
    plotter.add_mesh(mesh, scalars=property)        #type: ignore
    plotter.show()                                  #type: ignore


@allow_mesh
def save_mesh(mesh: DataSet, 
              property:str, 
              path:Path = DEFAULT_PATH, 
              filename:str|None=None,
              *,
              screenshot_size:int = SCREENSHOT_SIZE,) -> None:
    check_and_create_folder(path)
    filename = _default_filename(filename=filename, property=property)


    path = path/filename

    plotter = Plotter(off_screen=True)  
    plotter.add_mesh(mesh, scalars=property)                #type: ignore
    plotter.screenshot(filename=path,scale=screenshot_size) #type: ignore     


@allow_mesh
def slice_and_save(mesh: DataSet, 
                   property:str, 
                   normal: vector, 
                   *, 
                   slice_origin:vector=ORIGIN_VECTOR,
                   path:Path = DEFAULT_PATH, 
                   filename:str|None=None,
                   screenshot_size:int = SCREENSHOT_SIZE,
                   ) -> None:
    
    check_and_create_folder(path)
    filename = _default_filename(filename=filename, property=property)


    path = path/filename

    mesh=mesh.slice(normal=normal, origin=slice_origin)         #type: ignore

    plotter = pyvista.plotting.Plotter(off_screen=True)         #type: ignore
    plotter.add_mesh(mesh, scalars=property)                    #type: ignore

    plotter.enable_parallel_projection()                        #type: ignore
    plotter.camera_position = normal

    plotter.screenshot(filename=path, scale=screenshot_size)    #type: ignore

@allow_mesh
def xz_slice(mesh: DataSet, 
            property:str, 
            *, 
            slice_origin:vector=ORIGIN_VECTOR,
            path:Path = DEFAULT_PATH, 
            filename:str|None=None,
            screenshot_size:int = SCREENSHOT_SIZE,
                ) -> None:
    normal = PlaneNormals.XZ
    return slice_and_save(mesh,
                          property=property,
                          normal=normal,
                          slice_origin=slice_origin,
                          path=path,
                          filename=filename,
                          screenshot_size=screenshot_size)

@allow_mesh
def xy_slice(mesh: DataSet, 
            property:str, 
            *, 
            slice_origin:vector=ORIGIN_VECTOR,
            path:Path = DEFAULT_PATH, 
            filename:str|None=None,
            screenshot_size:int = SCREENSHOT_SIZE,
                ) -> None:
    normal = PlaneNormals.XY
    return slice_and_save(mesh,
                          property=property,
                          normal=normal,
                          slice_origin=slice_origin,
                          path=path,
                          filename=filename,
                          screenshot_size=screenshot_size)

@allow_mesh
def yz_slice(mesh: DataSet, 
            property:str, 
            *, 
            slice_origin:vector=ORIGIN_VECTOR,
            path:Path = DEFAULT_PATH, 
            filename:str|None=None,
            screenshot_size:int = SCREENSHOT_SIZE,
                ) -> None:
    normal = PlaneNormals.YZ
    return slice_and_save(mesh,
                          property=property,
                          normal=normal,
                          slice_origin=slice_origin,
                          path=path,
                          filename=filename,
                          screenshot_size=screenshot_size)

def glob_properties(input: Simulation| Mesh| SimulationPreprocessing|SimulationResults|ExtractedDataFields|NumericalResults|list[ParticleDetector]|list[Mesh] ,
                    property:str, *, ignore_num_kernel:bool=True ) -> list[tuple[Mesh, "str"]]:
    
    
    result:list[tuple[Mesh, "str"]]  = []
    if isinstance(input, Mesh):
        strings = fnmatch.filter(input.properties, property)
        for i in strings:
            result.append((input,i))
        return result

    if isinstance(input, Simulation):
        result += glob_properties(input.preprocessing, property=property)
        result += glob_properties(input.results, property=property)

    if isinstance(input, SimulationPreprocessing):
        result += glob_properties(input.model, property=property)

    if isinstance(input, SimulationResults):
        result += glob_properties(input.extracted_data_fields, property=property)
        if not ignore_num_kernel:
            result += glob_properties(input.numerical_kernel_output, property=property)

    if isinstance(input, NumericalResults):
        result += glob_properties(input.particle_detectors, property=property)

    if isinstance(input, list):
        for i in input:
            if isinstance(i, ParticleDetector):
                result += glob_properties(i.differential_flux_mesh, property=property)
                result += glob_properties(i.initial_distribution_mesh, property=property)
                result += glob_properties(i.distribution_function_mesh, property=property)
            if isinstance(i, Mesh):
                result += glob_properties(i, property=property)
                

    if isinstance(input, ExtractedDataFields):
        result += glob_properties(input.spacecraft_face, property=property)
        result += glob_properties(input.spacecraft_mesh, property=property)
        result += glob_properties(input.spacecraft_vertex, property=property)
        result += glob_properties(input.volume_vertex, property=property)
    
    return result
