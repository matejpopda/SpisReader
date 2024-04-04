from spisModule.reader import Simulation
from pathlib import Path
import pyvista.plotting
import pyvista.core as pvc
import spisModule.reader as reader
from typing import Callable
from pyvista.core.dataset import DataSet
import logging as log

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







def interactive_plot_orth_slice(mesh: DataSet|reader.Mesh, property:str) -> None:
    if isinstance(mesh, reader.Mesh):
        mesh = mesh.mesh

    interactive_plot_mesh(mesh.slice_orthogonal(), property=property) # type: ignore


def interactive_plot_physical_mesh(mesh: DataSet|reader.Mesh) -> None:
    if isinstance(mesh, reader.Mesh):
        mesh = mesh.mesh

    interactive_plot_mesh(mesh, "gmsh:physical")



def interactive_plot_mesh(mesh: DataSet|reader.Mesh, property:str) -> None:
    if isinstance(mesh, reader.Mesh):
        mesh = mesh.mesh

    plotter = pyvista.plotting.Plotter() # type: ignore
    plotter.add_mesh(mesh, scalars=property)  # type: ignore
    plotter.show()     # type: ignore


def save_mesh(mesh: DataSet|reader.Mesh, property:str, path:Path = Path("./temp"), filename:str|None=None) -> None:
    if isinstance(mesh, reader.Mesh):
        mesh = mesh.mesh

    if not path.exists(): 
        log.info(f"Output folder {str(path.resolve())} does not exist, creating it")
        path.mkdir()   

    if filename is None:
        filename = property + ".png"

    path = path/filename

    plotter = pyvista.plotting.Plotter(off_screen=True) # type: ignore
    plotter.add_mesh(mesh, scalars=property)  # type: ignore
    plotter.screenshot(filename=path)     # type: ignore
