# TODO
- add MSHtoVTK library in order to keep physical groups
- better access to deeply nested classes
- load trajectories
- Clean properties function - It will remove loaded data and move the property back into unloaded properties 
- Particle detector is the last class that takes up lot of memory
- Singleton for setting default settings and paths
- Make a function for drawing timeseries
- Function for writing out the moments
- Some sort of check to see which files didnt get loaded at all (with optional ignore list)
- Logging file next to the saved simulation, mainly with errors.

## Maybe
- In get_extracted_datafields() save the timeseries
- Save monitored_datafields

## Done
- Add properties to the Simulation class for easier navigation


# FIX
- reading of BC and SC
- use pyvista.DataSetFilters.extract_cells for when loading particle detector SC and BC current injections


# Basic use

## Loading a simulation
From the reader.py import load_simulation. Using this function you can load an output from SPIS by specifying path to the folder that has the SPIS study in it. 

For the path argument you should use pathlib, for example pathlib.Path("C:/data/example.spis5/S01_01").

You can also specify the name of the saved pickle file. Currently it gets saved next to the study directory.

Lastly you can force reloading of the dataset (for example after an update) by setting force_processing to True.

## Simulation class
In the simulation.py the dataclass Simulation is declared. All the data from the SPIS data folder gets saved in it. The structure of the class copies the structure of the SPIS folders with few naming changes. 

Furthermore there are few properties declared that allow easier navigation.

## Helpers.py
The helpers file contains a function default_log_config(). If you want to use logging, this function should be called at the beginning of the program.

This sets up logging for the program, where all logs get saved into a file called latest_runs.log and messages of level INFO and higher get output to the console.

## Plotting.py
All functions for plotting are declared in this file. 

Usually the first argument for these functions is the mesh we want to plot onto, the second argument is usually the property we want to plot, the names are taken from the SPIS files.

For functions that save images, there is a named argument "path" for setting where the images should be saved. There is also a named argument "filename", if the filename is not provided the property name is used.

### Retrieving properties by name
Use function glob_properties(simulation, property) to recursively search the simulation for a property. It allows UNIX-like wildcards, such as * and ?. 

It returns list of pairs (mesh, property) where the mesh contains the property and thus can be plotted. 

The advantage of using this function is that you are guaranteed that the property exists on the mesh that gets returned.

Output of this function can also be directly fed into the functions that make animated plots.


### Interactive plots
There are 3 interactive functions, calling them opens the interactive pyvista window. These are interactive_plot_orth_slice(), interactive_plot_physical_mesh(), 
interactive_plot_mesh()

### Image plots
For saving images there is:
- save_mesh() - makes an image from the default location
- slice_and_save() - it makes a 2d image after slicing an image using a normal vector, the origin of the slice can also be specified
- xz_slice(), xy_slice(), yz_slice() - helper functions that calls the above function with specified normal. 

The file also contains two read-only classes called Directions and PlaneNormals which contain named normal vectors.  

### Animated plots


- make_gif_xz_slice() - Makes the list of (mesh, property) into xz slices and plots it into a gif 
- make_gif_surface_from_default_view() - Makes the list of (mesh, property) into a gif





### Wrappers

Function plot_final_quantities() plots all final quantities into one folder.