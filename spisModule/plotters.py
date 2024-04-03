from spisModule.reader import Simulation
from pathlib import Path
import pyvista.plotting
import pyvista.core as pvc

def interactive_plot_physical_mesh(mesh: pvc.dataset.DataSet):
    interactive_plot_mesh(mesh, "gmsh:physical")



def interactive_plot_mesh(mesh: pvc.dataset.DataSet, property:str) -> None:
    plotter = pyvista.plotting.Plotter() # type: ignore
    plotter.add_mesh(mesh, scalars=property)  # type: ignore
    plotter.show()     # type: ignore







def save_all_particle_densities(simulation: Simulation, output: Path) -> None:
    # This function goes through all 
    pass