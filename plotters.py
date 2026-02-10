import collections.abc
from pathlib import Path
import pyvista.core.pyvista_ndarray
import pyvista.plotting
from pyvista.plotting.plotter import Plotter
from pyvista.core.dataset import DataSet
import pyvista.utilities
import pyvista.core.utilities.points
import pyvista.utilities
from simulation import *
import collections
import pyvista.core.dataset
from helpers import allow_mesh, check_and_create_folder
import logging
from dataclasses import dataclass
import fnmatch
import numpy as np
import numpy.typing as np_typing
from default_settings import Settings
import utils
import matplotlib.pyplot as plt
import scipy.constants
import random
from typing import Tuple
import electron_detector
from electron_detector import CollisionTypes

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
def interactive_plot_mesh_with_trajectories(mesh: DataSet, trajectories: list[list[vector]]) -> None:
    plotter = Plotter()
    plotter.add_mesh(mesh, scalars="gmsh:physical")  # type: ignore

    for trajectory in trajectories:
        # print(trajectory)
        line = pyvista.core.utilities.points.lines_from_points(trajectory)  # type: ignore
        plotter.add_mesh(line, color="black")  # type: ignore

    plotter.show()  # type: ignore


def plot_detectors_with_0_acceptance_angle(detectors: list[electron_detector.ElectronDetector]):
    for detector in detectors:
        energies = []
        ambient_probs = []
        sec_probs = []
        for particle in detector.result_accumulator.particles:
            if particle.probability_ambient is None:
                ambient_probs.append(None)
            else:
                ambient_probs.append(particle.probability_ambient)  # * 6.24e16

            if particle.probability_secondary is None:
                sec_probs.append(None)
            else:
                sec_probs.append(particle.probability_secondary)  # * 6.24e16

            energies.append(particle.starting_energy)
            ambient_probs.append(particle.probability_ambient * 6.24 * 10**18)
            sec_probs.append(particle.probability_secondary)

        plt.scatter(energies, ambient_probs, label=f"Detector {detector.backtracking_type.name} - Ambient")
        # plt.scatter(energies, sec_probs, label=f"Detector {detector.backtracking_type.name}  - Secondary")

    plt.xlabel("Energy [eV]")
    plt.ylabel("Velocity distribution function [$m^{-6} s^3$]")
    plt.title("Distribution function for particles moving away from the spacecraft")

    plt.legend()
    plt.show()


def plot_detectors_with_0_acceptance_angle_ev(detectors: list[electron_detector.ElectronDetector]):
    for detector in detectors:
        energies = []
        ambient_probs = []
        sec_probs = []
        for particle in detector.result_accumulator.particles:
            correction_factor = 1.0 / np.sqrt(2 * particle.MASS * particle.CHARGE * particle.starting_energy)
            if particle.probability_ambient is None:
                ambient_probs.append(None)
            else:
                ambient_probs.append(particle.probability_ambient * correction_factor)  # * 6.24e16

            if particle.probability_secondary is None:
                sec_probs.append(None)
            else:
                sec_probs.append(particle.probability_secondary * correction_factor)  # * 6.24e16

            energies.append(particle.starting_energy)

        plt.scatter(energies, ambient_probs, label=f"Detector {detector.backtracking_type.name} - Ambient")
        # plt.scatter(energies, sec_probs, label=f"Detector {detector.backtracking_type.name}  - Secondary")

    plt.xlabel("Energy [eV]")
    plt.ylabel("Velocity distribution function [$m^{-3} eV^{-1}$]")
    plt.title("Distribution function for particles moving away from the spacecraft")

    plt.legend()
    plt.show()


def plot_electron_potential_history(electron: electron_detector.Electron):
    x: list[float] = []
    dts: list[float] = []
    y: list[float] = []

    z: list[float] = []

    for i, (dt, energy, potential) in enumerate(
        zip(electron.dt_history, electron.energy_history, electron.potential_history)
    ):
        if i < 1:
            continue
        x.append(sum(dts) + dt)
        dts.append(dt)
        y.append(energy)
        z.append(potential)

    plt.scatter(x, y, label="Particles kinetic energy", c="blue")
    plt.plot(x, z, label="Plasma potential", c="black")
    plt.legend()
    plt.ylabel("Particle energy/ Plasma potential [eV]")
    plt.title("Comparison of plasma potential with particles kinetic energy")
    plt.xlabel("Elapsed time [s]")
    plt.show()


def detectors_to_1d_distribution_bad(detectors: list[electron_detector.ElectronDetector]):
    energy: list[float] = []
    p_ambient: list[float] = []
    p_seee: list[float] = []
    p_photo: list[float] = []
    p_total: list[float] = []

    def calculate_avg_probability(detector: electron_detector.ElectronDetector):
        result_ambient = 0
        result_seee = 0
        result_photo = 0
        result_total = 0

        for particle in detector.result_accumulator.particles:
            if particle.probability_ambient is not None:
                result_ambient += particle.probability_ambient
                result_total += particle.probability_ambient
            if particle.probability_secondary is not None:
                result_seee += particle.probability_secondary
                result_total += particle.probability_secondary
            if particle.probability_photo is not None:
                result_photo += particle.probability_photo
                result_total += particle.probability_photo

        p_ambient.append(result_ambient / len(detector.result_accumulator.particles))
        p_seee.append(result_seee / len(detector.result_accumulator.particles))
        p_photo.append(result_photo / len(detector.result_accumulator.particles))
        p_total.append(result_total / len(detector.result_accumulator.particles))

        # p_ambient.append(result_ambient)
        # p_seee.append(result_seee )
        # p_photo.append(result_photo )

    for detector in detectors:
        energy.append(detector.energy)
        calculate_avg_probability(detector)

    plt.scatter(energy, p_ambient, c="blue")
    plt.scatter(energy, p_seee, c="green")
    plt.scatter(energy, p_photo, c="orange")
    plt.scatter(energy, p_total, c="black")
    # plt.yscale('log')
    plt.xscale("log")
    plt.show()


def detectors_to_1d_distribution(detectors: list[electron_detector.ElectronDetector]):
    energy: list[float] = []
    p_ambient: list[float] = []
    p_seee: list[float] = []
    p_photo: list[float] = []
    p_total: list[float] = []

    def calculate_probability(detector: electron_detector.ElectronDetector):
        result_ambient = 0
        result_seee = 0
        result_photo = 0
        result_total = 0

        for particle in detector.result_accumulator.particles:
            particle.collision_type
            particle.origin  # phi and theta on a sphere

            if particle.probability_ambient is not None:
                result_ambient += particle.probability_ambient * np.cos(particle.origin[1])
                result_total += particle.probability_ambient * np.cos(particle.origin[1])
            if particle.probability_secondary is not None:
                result_seee += particle.probability_secondary * np.cos(particle.origin[1])
                result_total += particle.probability_secondary * np.cos(particle.origin[1])
            if particle.probability_photo is not None:
                result_photo += particle.probability_photo * np.cos(particle.origin[1])
                result_total += particle.probability_photo * np.cos(particle.origin[1])

        p_ambient.append(result_ambient)
        p_seee.append(result_seee)
        p_photo.append(result_photo)
        p_total.append(result_total)

    for detector in detectors:
        energy.append(detector.energy)
        calculate_probability(detector)

        # for i,j,k in zip(p_ambient, p_photo, p_seee):
        #     p_total.append(i+j+k)

    plt.scatter(energy, p_ambient, c="blue", label="Ambient electrons")
    plt.scatter(energy, p_seee, c="green", label="Secondary electrons")
    # plt.scatter(energy, p_photo, c="orange")
    plt.plot(energy, p_total, c="black", label="Total electrons")
    plt.axvline(x=13.9, color="r", linestyle="--", label="Spacecraft potential")

    plt.legend()

    plt.xlabel("Energy [eV]")
    plt.ylabel("Distribution function [$m^{-3} eV^{-1}$]")

    plt.title("EAS distribution function")

    plt.xscale("log")
    plt.yscale("log")
    plt.show()


@allow_mesh
def interactive_plot_mesh_with_typed_trajectories(
    mesh: DataSet, trajectories: list[Tuple[list[vector], CollisionTypes]]
) -> None:
    plotter = Plotter()
    plotter.add_mesh(mesh, scalars="gmsh:physical")  # type: ignore

    # plotter.add_mesh(pyvista.core.utilities.points.lines_from_points([[0,0,0], [-10,0,0]]), color="green")
    for trajectory in trajectories:
        # print(trajectory)

        line = pyvista.core.utilities.points.lines_from_points(trajectory[0])  # type: ignore
        color = "black"
        if trajectory[1] == CollisionTypes.Spacecraft:
            color = "red"
        if trajectory[1] == CollisionTypes.Boundary:
            color = "blue"
        plotter.add_mesh(line, color=color)  # type: ignore
    plotter.show()  # type: ignore


@allow_mesh
def interactive_plot_electron_detectors(
    mesh: DataSet, detectors: list[electron_detector.ElectronDetector]
) -> None:
    plotter = Plotter()
    plotter.add_mesh(mesh, scalars="gmsh:physical")  # type: ignore

    # plotter.add_mesh(pyvista.core.utilities.points.lines_from_points([[0,0,0], [-10,0,0]]), color="green")
    for detector in detectors:
        # print(trajectory)
        for particle in detector.result_accumulator.particles:
            # if not (particle.probability_photo is not None and particle.probability_photo > 0):
            #     continue

            # if particle.position[0] > -1 or particle.collision_type == CollisionTypes.Boundary:
            #     continue
            # if not particle.collision_type == CollisionTypes.Spacecraft:
            #     continue

            line = pyvista.core.utilities.points.lines_from_points(particle.position_history)  # type: ignore
            color = "black"
            if particle.collision_type == CollisionTypes.Spacecraft:
                color = "red"
            if particle.collision_type == CollisionTypes.Boundary:
                color = "blue"
            if particle.probability_photo is not None and particle.probability_photo > 0:
                color = "yellow"
            plotter.add_mesh(line, color=color)  # type: ignore
    plotter.show()  # type: ignore@allow_mesh


@allow_mesh
def interactive_plot_electron_detectors_differentiate_detectors_by_color(
    mesh: DataSet, detectors: list[electron_detector.ElectronDetector]
) -> None:
    plotter = Plotter()
    plotter.add_mesh(mesh, scalars="gmsh:physical")  # type: ignore

    # plotter.add_mesh(pyvista.core.utilities.points.lines_from_points([[0,0,0], [-10,0,0]]), color="green")
    for detector in detectors:
        # print(trajectory)
        for particle in detector.result_accumulator.particles:
            # if not (particle.probability_photo is not None and particle.probability_photo > 0):
            #     continue

            # if particle.position[0] > -1 or particle.collision_type == CollisionTypes.Boundary:
            #     continue
            # if not particle.collision_type == CollisionTypes.Spacecraft:
            #     continue

            line = pyvista.core.utilities.points.lines_from_points(particle.position_history)  # type: ignore
            color = "black"
            if detector.backtracking_type == electron_detector.BacktrackingTypes.Euler:
                color = "red"
            if detector.backtracking_type == electron_detector.BacktrackingTypes.RK:
                color = "blue"
            if detector.backtracking_type == electron_detector.BacktrackingTypes.Boris:
                color = "orange"
            plotter.add_mesh(line, color=color)  # type: ignore
    plotter.show()  # type: ignore@allow_mesh


@allow_mesh
def interactive_plot_electron_detectors_XY(
    mesh: DataSet, detectors: list[electron_detector.ElectronDetector]
) -> None:
    plotter = Plotter()
    plotter.add_mesh(mesh, scalars="gmsh:physical")  # type: ignore

    # plotter.add_mesh(pyvista.core.utilities.points.lines_from_points([[0,0,0], [-10,0,0]]), color="green")
    for detector in detectors:
        # print(trajectory)
        for particle in detector.result_accumulator.particles:
            if not (0 < (particle.origin[1]) < 0.1):
                continue

            # if not (particle.probability_photo is not None and particle.probability_photo > 0):
            #     continue

            # if particle.position[0] > -1 or particle.collision_type == CollisionTypes.Boundary:
            #     continue
            # if not particle.collision_type == CollisionTypes.Spacecraft:
            #     continue

            line = pyvista.core.utilities.points.lines_from_points(particle.position_history)  # type: ignore
            color = "black"
            if detector.backtracking_type == electron_detector.BacktrackingTypes.Euler:
                color = "red"
            if detector.backtracking_type == electron_detector.BacktrackingTypes.RK:
                color = "green"
            if detector.backtracking_type == electron_detector.BacktrackingTypes.Boris:
                color = "blue"
            plotter.add_mesh(line, color=color)  # type: ignore
    plotter.show()  # type: ignore@allow_mesh


@allow_mesh
def interactive_plot_electron_detectors_XZ(
    mesh: DataSet, detectors: list[electron_detector.ElectronDetector]
) -> None:
    plotter = Plotter()
    plotter.add_mesh(mesh, scalars="gmsh:physical")  # type: ignore

    # plotter.add_mesh(pyvista.core.utilities.points.lines_from_points([[0,0,0], [-10,0,0]]), color="green")
    for detector in detectors:
        # print(trajectory)
        for particle in detector.result_accumulator.particles:
            if not (0 < (particle.origin[0]) < 0.1):
                continue

            # if not (particle.probability_photo is not None and particle.probability_photo > 0):
            #     continue

            # if particle.position[0] > -1 or particle.collision_type == CollisionTypes.Boundary:
            #     continue
            # if not particle.collision_type == CollisionTypes.Spacecraft:
            #     continue

            line = pyvista.core.utilities.points.lines_from_points(particle.position_history)  # type: ignore
            color = "black"
            if detector.backtracking_type == electron_detector.BacktrackingTypes.Euler:
                color = "red"
            if detector.backtracking_type == electron_detector.BacktrackingTypes.RK:
                color = "green"
            if detector.backtracking_type == electron_detector.BacktrackingTypes.Boris:
                color = "blue"
            plotter.add_mesh(line, color=color)  # type: ignore
    plotter.show()  # type: ignore@allow_mesh


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
    percentile: float
    | None
    | typing.Sequence[float] = 0.05,  # TODO RENAME to better reflect option of adding a sequence
    view_up: vector | None = None,
    component: int | None = None,
) -> None:
    if path is None:
        path = Settings.default_output_path

    if screenshot_size is None:
        screenshot_size = Settings.screenshot_size

    check_and_create_folder(path)
    filename = _default_filename(filename=filename, property=property)

    path = path / filename

    if percentile is not None:
        if isinstance(percentile, collections.abc.Sequence):
            clim = (percentile[0], percentile[1])
        else:
            clim = _calculate_percentile(mesh, property=property, percentile=percentile)
    else:
        clim = None

    plotter = pyvista.plotting.Plotter(off_screen=True)  # type: ignore
    mesh = mesh.slice(normal=normal, origin=slice_origin)  # type:ignore
    plotter.add_mesh(mesh, scalars=property, clim=clim, component=component)  # type: ignore

    plotter.enable_parallel_projection()  # type: ignore
    plotter.camera_position = [normal, slice_origin, (0, 1, 0)]

    if view_up is not None:
        plotter.set_viewup(view_up)  # type: ignore

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
    component: int | None = None,
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
        view_up=PlaneNormals.XY,
        component=component,
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
    component: int | None = None,
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
        view_up=PlaneNormals.XZ_flipped,
        component=component,
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
    component: int | None = None,
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
        view_up=PlaneNormals.XZ,
        component=component,
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
    return utils.glob_properties(
        input=input, property=property, ignore_num_kernel=ignore_num_kernel, exclude=exclude
    )


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
