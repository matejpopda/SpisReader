from pathlib import Path
import pyvista.plotting
from pyvista.plotting.plotter import Plotter
from pyvista.core.dataset import DataSet
from simulation import *
import pyvista.core.dataset
from helpers import allow_mesh, check_and_create_folder
import logging
from dataclasses import dataclass
import math
import fnmatch

log = logging.getLogger(__name__)


SCREENSHOT_SIZE = 2

ORIGIN_VECTOR = (0, 0, 0)
DEFAULT_PATH = Path("./temp")


@dataclass(frozen=True)
class Directions:
    x_plus = (1, 0, 0)
    x_minus = (-1, 0, 0)
    y_plus = (0, 1, 0)
    y_minus = (0, -1, 0)
    z_plus = (0, 0, 1)
    z_minus = (0, 0, -1)


@dataclass(frozen=True)
class PlaneNormals:
    XY = (0, 0, 1)
    XZ = (0, 1, 0)
    YZ = (1, 0, 0)
    XY_flipped = (0, 0, -1)
    XZ_flipped = (0, -1, 0)
    YZ_flipped = (-1, 0, 0)


vector = Directions | PlaneNormals | tuple[float, float, float] | tuple[int, int, int]


def _default_filename(filename: str | None, property: str) -> str:
    if filename is None:
        filename = property + ".png"
    return filename


@allow_mesh
def interactive_plot_orth_slice(mesh: DataSet, property: str) -> None:
    temp: DataSet = mesh.slice_orthogonal()  # type: ignore
    interactive_plot_mesh(temp, property=property)  # type: ignore


@allow_mesh
def interactive_plot_physical_mesh(mesh: DataSet) -> None:
    interactive_plot_mesh(mesh, "gmsh:physical")


@allow_mesh
def interactive_plot_mesh(mesh: DataSet, property: str) -> None:
    plotter = Plotter()
    plotter.add_mesh(mesh, scalars=property)  # type: ignore
    plotter.show()  # type: ignore


@allow_mesh
def save_mesh(
    mesh: DataSet,
    property: str,
    path: Path = DEFAULT_PATH,
    filename: str | None = None,
    *,
    screenshot_size: int = SCREENSHOT_SIZE,
    clim: tuple[float, float] | None = None,
) -> None:
    check_and_create_folder(path)
    filename = _default_filename(filename=filename, property=property)

    path = path / filename

    plotter = Plotter(off_screen=True)
    plotter.add_mesh(mesh, scalars=property, clim=clim)  # type: ignore
    plotter.screenshot(filename=path, scale=screenshot_size)  # type: ignore


@allow_mesh
def slice_and_save(
    mesh: DataSet,
    property: str,
    normal: vector,
    *,
    slice_origin: vector = ORIGIN_VECTOR,
    path: Path = DEFAULT_PATH,
    filename: str | None = None,
    screenshot_size: int = SCREENSHOT_SIZE,
    clim: tuple[float, float] | None = None,
) -> None:
    check_and_create_folder(path)
    filename = _default_filename(filename=filename, property=property)

    path = path / filename

    plotter = pyvista.plotting.Plotter(off_screen=True)  # type: ignore
    mesh = mesh.slice(normal=PlaneNormals.XZ, origin=slice_origin)  # type:ignore
    plotter.add_mesh(mesh, scalars=property, clim=clim)  # type: ignore

    plotter.enable_parallel_projection()  # type: ignore
    plotter.camera_position = normal

    plotter.screenshot(filename=path, scale=screenshot_size)  # type: ignore


@allow_mesh
def xz_slice(
    mesh: DataSet,
    property: str,
    *,
    slice_origin: vector = ORIGIN_VECTOR,
    path: Path = DEFAULT_PATH,
    filename: str | None = None,
    screenshot_size: int = SCREENSHOT_SIZE,
    clim: tuple[float, float] | None = None,
) -> None:
    normal = PlaneNormals.XZ
    return slice_and_save(
        mesh,
        property=property,
        normal=normal,
        slice_origin=slice_origin,
        path=path,
        filename=filename,
        screenshot_size=screenshot_size,
        clim=clim,
    )


@allow_mesh
def xy_slice(
    mesh: DataSet,
    property: str,
    *,
    slice_origin: vector = ORIGIN_VECTOR,
    path: Path = DEFAULT_PATH,
    filename: str | None = None,
    screenshot_size: int = SCREENSHOT_SIZE,
    clim: tuple[float, float] | None = None,
) -> None:
    normal = PlaneNormals.XY
    return slice_and_save(
        mesh,
        property=property,
        normal=normal,
        slice_origin=slice_origin,
        path=path,
        filename=filename,
        screenshot_size=screenshot_size,
        clim=clim,
    )


@allow_mesh
def yz_slice(
    mesh: DataSet,
    property: str,
    *,
    slice_origin: vector = ORIGIN_VECTOR,
    path: Path = DEFAULT_PATH,
    filename: str | None = None,
    screenshot_size: int = SCREENSHOT_SIZE,
    clim: tuple[float, float] | None = None,
) -> None:
    normal = PlaneNormals.YZ
    return slice_and_save(
        mesh,
        property=property,
        normal=normal,
        slice_origin=slice_origin,
        path=path,
        filename=filename,
        screenshot_size=screenshot_size,
        clim=clim,
    )


def glob_properties(
    input: (
        Simulation
        | Mesh
        | SimulationPreprocessing
        | SimulationResults
        | ExtractedDataFields
        | NumericalResults
        | list[ParticleDetector]
        | list[Mesh]
    ),
    property: str,
    *,
    ignore_num_kernel: bool = True,
    exclude: str | None = None,
) -> list[tuple[Mesh, "str"]]:
    result: list[tuple[Mesh, "str"]] = []
    if isinstance(input, Mesh):
        strings = fnmatch.filter(input.properties, property)
        for i in strings:
            if exclude is None or not fnmatch.fnmatch(i, exclude):
                result.append((input, i))
        return result

    if isinstance(input, Simulation):
        result += glob_properties(
            input.preprocessing,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )
        result += glob_properties(
            input.results,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )

    if isinstance(input, SimulationPreprocessing):
        result += glob_properties(
            input.model,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )

    if isinstance(input, SimulationResults):
        result += glob_properties(
            input.extracted_data_fields,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )
        if not ignore_num_kernel:
            result += glob_properties(
                input.numerical_kernel_output,
                property=property,
                ignore_num_kernel=ignore_num_kernel,
                exclude=exclude,
            )

    if isinstance(input, NumericalResults):
        result += glob_properties(
            input.particle_detectors,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )

    if isinstance(input, list):
        for i in input:
            if isinstance(i, ParticleDetector):
                result += glob_properties(
                    i.differential_flux_mesh,
                    property=property,
                    ignore_num_kernel=ignore_num_kernel,
                    exclude=exclude,
                )
                result += glob_properties(
                    i.initial_distribution_mesh,
                    property=property,
                    ignore_num_kernel=ignore_num_kernel,
                    exclude=exclude,
                )
                result += glob_properties(
                    i.distribution_function_mesh,
                    property=property,
                    ignore_num_kernel=ignore_num_kernel,
                    exclude=exclude,
                )
            if isinstance(i, Mesh):
                result += glob_properties(
                    i,
                    property=property,
                    ignore_num_kernel=ignore_num_kernel,
                    exclude=exclude,
                )

    if isinstance(input, ExtractedDataFields):
        result += glob_properties(
            input.spacecraft_face,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )
        result += glob_properties(
            input.spacecraft_mesh,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )
        result += glob_properties(
            input.spacecraft_vertex,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )
        result += glob_properties(
            input.volume_vertex,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )
        result += glob_properties(
            input.display_vol_mesh,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )

    return result


def make_gif_xz_slice(
    input: list[tuple[Mesh, "str"]],
    filename: str,
    *,
    slice_origin: vector = ORIGIN_VECTOR,
    path: Path = DEFAULT_PATH,
    screenshot_size: int = SCREENSHOT_SIZE,
) -> None:
    if not fnmatch.fnmatch(filename, "*.gif"):
        filename = filename + ".gif"
    plotter = Plotter(off_screen=True)

    plotter.open_gif(str(path / (filename)))  # type: ignore
    plotter.window_size = [plotter.window_size[0] * SCREENSHOT_SIZE, plotter.window_size[1] * SCREENSHOT_SIZE]  # type: ignore

    min_val, max_val = math.inf, -math.inf
    for mesh, property in input:
        cur_min, cur_max = mesh.mesh.get_data_range(property)  # type: ignore
        min_val = min(cur_min, min_val)
        max_val = max(cur_max, max_val)

    for mesh, property in input:
        mesh = mesh.mesh.slice(
            normal=PlaneNormals.XZ, origin=slice_origin
        )  # type:ignore
        plotter.add_mesh(mesh, scalars=property, clim=(min_val, max_val), style="surface")  # type: ignore
        plotter.enable_parallel_projection()  # type: ignore
        plotter.camera_position = PlaneNormals.XZ
        plotter.write_frame()
        plotter.clear()
    plotter.close()


def make_gif_surface_from_default_view(
    input: list[tuple[Mesh, "str"]],
    filename: str,
    *,
    slice_origin: vector = ORIGIN_VECTOR,
    path: Path = DEFAULT_PATH,
    screenshot_size: int = SCREENSHOT_SIZE,
) -> None:
    if not fnmatch.fnmatch(filename, "*.gif"):
        filename = filename + ".gif"
    plotter = Plotter(off_screen=True)

    plotter.open_gif(str(path / (filename)))  # type: ignore
    plotter.window_size = [plotter.window_size[0] * SCREENSHOT_SIZE, plotter.window_size[1] * SCREENSHOT_SIZE]  # type: ignore

    min_val, max_val = math.inf, -math.inf
    for mesh, property in input:
        cur_min, cur_max = mesh.mesh.get_data_range(property)  # type: ignore
        min_val = min(cur_min, min_val)
        max_val = max(cur_max, max_val)

    for mesh, property in input:
        plotter.add_mesh(mesh.mesh, scalars=property, clim=(min_val, max_val))  # type: ignore
        plotter.enable_parallel_projection()  # type: ignore
        plotter.write_frame()
        plotter.clear()
    plotter.close()


def plot_final_quantities(result: Simulation, path: Path = DEFAULT_PATH):
    log.info("Started plotting final quantities")
    for i, j in glob_properties(result, "*final*", exclude="*surf*"):
        xz_slice(i, j, path=path)
    log.info("Done plotting final quantities")
