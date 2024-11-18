from pathlib import Path
import pyvista.core.pyvista_ndarray
import pyvista.plotting
from pyvista.plotting.plotter import Plotter
from pyvista.core.dataset import DataSet
from simulation import *
import pyvista.core.dataset
from helpers import allow_mesh, check_and_create_folder
import logging
from dataclasses import dataclass
import fnmatch
import numpy as np
import numpy.typing as np_typing
from default_settings import Settings
import reader


log = logging.getLogger(__name__)


ORIGIN_VECTOR = (0, 0, 0)


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


def _calculate_percentile(
    meshes: list[DataSet] | list[tuple[DataSet, str]] | DataSet | list[tuple[Mesh, str]],
    property: str | None = None,
    percentile: float | None = None,
) -> tuple[float, float]:
    result: list[tuple[DataSet, str]] = []

    if percentile is None:
        percentile = Settings.percentile

    if isinstance(meshes, DataSet):
        if property is None:
            raise RuntimeError(f"Didn't specify property")
        result = [(meshes, property)]
    else:
        if isinstance(meshes[0], DataSet):
            if property is None:
                raise RuntimeError(f"Didn't specify property")
            for mesh in meshes:
                assert isinstance(mesh, DataSet)
                result.append((mesh, property))
        elif isinstance(meshes[0], tuple):  # is tuple # type:ignore
            for i in meshes:
                assert not isinstance(i, DataSet)

            if all(not isinstance(x, DataSet) and isinstance(x[0], Mesh) for x in meshes):
                for mesh, local_property in meshes:  # type: ignore
                    assert isinstance(local_property, str)
                    assert isinstance(mesh, Mesh)
                    result.append((mesh.mesh, local_property))
            elif all(not isinstance(x, DataSet) and isinstance(x[0], DataSet) for x in meshes):
                result = meshes  # type:ignore
            else:
                raise RuntimeError("Unknown input")
        else:
            raise RuntimeError("Unknown input")

    all_values: np_typing.NDArray[np.float_] = np.array([])
    for mesh, local_property in result:
        array: np_typing.NDArray[np.float_] = np.array(mesh.get_array(local_property))  # type: ignore
        all_values = np.concatenate((all_values, array), axis=None)  # type: ignore

    temp = (
        np.percentile(all_values, int(percentile * 100)),
        np.percentile(all_values, abs(int(percentile * 100) - 100)),
    )

    return (min(temp), max(temp))  # type: ignore


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
    path: Path | None = None,
    filename: str | None = None,
    *,
    screenshot_size: float | None = None,
    clim: tuple[float, float] | None = None,
) -> None:
    if screenshot_size is None:
        screenshot_size = Settings.screenshot_size

    if path is None:
        path = Settings.default_output_path
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
    path: Path | None = None,
    filename: str | None = None,
    screenshot_size: float | None = None,
    percentile: float | None = 0.05,
    view_up: vector|None = None,
) -> None:
    if path is None:
        path = Settings.default_output_path

    if screenshot_size is None:
        screenshot_size = Settings.screenshot_size

    check_and_create_folder(path)
    filename = _default_filename(filename=filename, property=property)

    path = path / filename

    if percentile is not None:
        clim = _calculate_percentile(mesh, property=property, percentile=percentile)
    else:
        clim = None

    plotter = pyvista.plotting.Plotter(off_screen=True)  # type: ignore
    mesh = mesh.slice(normal=normal, origin=slice_origin)  # type:ignore
    plotter.add_mesh(mesh, scalars=property, clim=clim)  # type: ignore

    plotter.enable_parallel_projection()  # type: ignore
    plotter.camera_position = normal

    if view_up is not None:
        plotter.set_viewup(view_up) # type: ignore

    plotter.screenshot(filename=path, scale=screenshot_size)  # type: ignore


@allow_mesh
def xz_slice(
    mesh: DataSet,
    property: str,
    *,
    slice_origin: vector = ORIGIN_VECTOR,
    path: Path | None = None,
    filename: str | None = None,
    screenshot_size: float | None = None,
    percentile: float | None = None,
) -> None:
    if path is None:
        path = Settings.default_output_path
    if percentile is None:
        percentile = Settings.percentile
    if screenshot_size is None:
        screenshot_size = Settings.screenshot_size

    normal = PlaneNormals.XZ
    return slice_and_save(
        mesh,
        property=property,
        normal=normal,
        slice_origin=slice_origin,
        path=path,
        filename=filename,
        screenshot_size=screenshot_size,
        percentile=percentile,
        view_up=PlaneNormals.XY
    )


@allow_mesh
def xy_slice(
    mesh: DataSet,
    property: str,
    *,
    slice_origin: vector = ORIGIN_VECTOR,
    path: Path | None = None,
    filename: str | None = None,
    screenshot_size: float | None = None,
    percentile: float | None = None,
) -> None:
    if path is None:
        path = Settings.default_output_path
    if percentile is None:
        percentile = Settings.percentile
    if screenshot_size is None:
        screenshot_size = Settings.screenshot_size

    normal = PlaneNormals.XY
    return slice_and_save(
        mesh,
        property=property,
        normal=normal,
        slice_origin=slice_origin,
        path=path,
        filename=filename,
        screenshot_size=screenshot_size,
        percentile=percentile,
        view_up=PlaneNormals.XZ_flipped
    )


@allow_mesh
def yz_slice(
    mesh: DataSet,
    property: str,
    *,
    slice_origin: vector = ORIGIN_VECTOR,
    path: Path | None = None,
    filename: str | None = None,
    screenshot_size: float | None = None,
    percentile: float | None = None,
) -> None:
    if percentile is None:
        percentile = Settings.percentile
    if path is None:
        path = Settings.default_output_path
    if screenshot_size is None:
        screenshot_size = Settings.screenshot_size
    normal = PlaneNormals.YZ
    return slice_and_save(
        mesh,
        property=property,
        normal=normal,
        slice_origin=slice_origin,
        path=path,
        filename=filename,
        screenshot_size=screenshot_size,
        percentile=percentile,
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
        unloaded_keys = fnmatch.filter(input.loadable_properties.keys(), property)
        if len(unloaded_keys) != 0:
            for key in unloaded_keys:
                reader.load_property_into_mesh(input, input.loadable_properties.pop(key))

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
    path: Path | None = None,
    screenshot_size: float | None = None,
    percentile: float | None = None,
) -> None:
    if path is None:
        path = Settings.default_output_path
    if percentile is None:
        percentile = Settings.percentile
    if screenshot_size is None:
        screenshot_size = Settings.screenshot_size

    if not fnmatch.fnmatch(filename, "*.gif"):
        filename = filename + ".gif"
    plotter = Plotter(off_screen=True)

    plotter.open_gif(str(path / (filename)))  # type: ignore
    plotter.window_size = [plotter.window_size[0] * screenshot_size, plotter.window_size[1] * screenshot_size]  # type: ignore

    min_val, max_val = _calculate_percentile(input)

    for mesh, property in input:
        cur_min, cur_max = mesh.mesh.get_data_range(property)  # type: ignore
        min_val = min(cur_min, min_val)
        max_val = max(cur_max, max_val)

    for mesh, property in input:
        mesh = mesh.mesh.slice(normal=PlaneNormals.XZ, origin=slice_origin)  # type:ignore
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
    path: Path | None = None,
    screenshot_size: float | None = None,
) -> None:
    if path is None:
        path = Settings.default_output_path
    if screenshot_size is None:
        screenshot_size = Settings.screenshot_size

    if not fnmatch.fnmatch(filename, "*.gif"):
        filename = filename + ".gif"
    plotter = Plotter(off_screen=True)

    plotter.open_gif(str(path / (filename)))  # type: ignore
    plotter.window_size = [plotter.window_size[0] * screenshot_size, plotter.window_size[1] * screenshot_size]  # type: ignore

    min_val, max_val = _calculate_percentile(input)

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


def plot_final_quantities(result: Simulation, path: Path | None = None, *, percentile: float | None = None):
    log.info("Started plotting final quantities")
    if percentile is None:
        percentile = Settings.percentile

    if path is None:
        path = Settings.default_output_path

    for i, property in glob_properties(result, "*final*", exclude="*surf*"):
        filename: str = property + ".png"

        xz_slice(i, property, path=path, percentile=percentile, filename="XZ_" + filename)

        xy_slice(i, property, path=path, percentile=percentile, filename="XY_" + filename)
        yz_slice(i, property, path=path, percentile=percentile, filename="YZ_" + filename)
    log.info("Done plotting final quantities")
