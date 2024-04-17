from pathlib import Path
import xml.etree.ElementTree as ET
import logging
import typing
import meshio
import meshio._mesh
import pyvista
import pyvista.core.dataset
import pickle
import pandas
import numpy as np
import xarray
import sparse
import numpy.typing
from helpers import LogFileOpening
from simulation import *

log = logging.getLogger(__name__)


def load_data(path: Path) -> Simulation:
    return Simulation(
        results=get_simulation_results(path / "Simulations" / "Run1"),
        preprocessing=get_simulation_preprocessing(path / "Preprocessing"),
    )


def get_simulation_preprocessing(
    path_to_preprocessing: Path,
) -> SimulationPreprocessing:
    model_path: list[Path] = list((path_to_preprocessing / "Mesh" / "GeometricalSystem").glob("*"))
    assert len(model_path) == 1
    model: Mesh = load_mesh(model_path[0])
    # Loads file Preprocessing/Mesh/GeometricalSystem/*.msh

    groups: list[Group] = get_groups(path_to_preprocessing / "Groups" / "groups.xml")
    # \Preprocessing\Groups/groups.xml

    electrical_circuit: str = (path_to_preprocessing / "ElectricalCircuit" / "circuit.txt").read_text()
    # \Preprocessing\ElectricalCircuit\circuit.txt

    return SimulationPreprocessing(model=model, groups=groups, electrical_circuit=electrical_circuit)


def get_simulation_results(path_to_results: Path) -> SimulationResults:
    default_instruments: list[DefaultInstrument] = get_default_instruments(
        path_to_results / "DefaultInstruments"
    )
    # List of instruments from \Simulations\Run1\DefaultInstruments

    user_instruments: list[UserInstrument] = get_user_instruments(path_to_results / "UserInstruments")
    # List of instruments from \Simulations\Run1\UserInstruments

    global_parameters = dictionary_from_list_in_xml_node(
        ET.parse(path_to_results / "GlobalParameters" / "globalParameters.xml").getroot()
    )
    # \Simulations\Run1\GlobalParameters

    numerical_kernel_output: NumericalResults = get_numerical_kernel_output(
        path_to_results / "NumKernel" / "Output", user_instruments
    )
    # \Simulations\Run1\NumKernel\Output

    monitored_data_fields = None  # TODO I think the data can be more easily gotten from numerical kernel

    extracted_data_fields: ExtractedDataFields = get_extracted_datafields(
        path_to_results / "OutputFolder" / "DataFieldExtracted"
    )
    # \Simulations\Run1\OutputFolder\DataFieldExtracted

    return SimulationResults(
        default_instruments=default_instruments,
        user_instruments=user_instruments,
        global_parameters=global_parameters,
        numerical_kernel_output=numerical_kernel_output,
        monitored_data_fields=monitored_data_fields,
        extracted_data_fields=extracted_data_fields,
    )


@LogFileOpening
def get_groups(path: Path) -> list[Group]:
    if not path.exists():
        raise FileNotFoundError("Missing file " + str(path))

    result: list[Group] = []

    tree = ET.parse(path)
    groupList = tree.getroot()[0]
    for children in groupList:
        if children.find("type") is not None:
            result.append(
                Group(
                    name=get_text_of_a_child(children, "name"),
                    SPISid=get_text_of_a_child(children, "id"),
                    GMSHid=get_text_of_a_child(get_child(children, "linkedMeshMaskIds"), "int"),
                    type=get_text_of_a_child(children, "type"),
                    properties=parsePropertiesList(get_child(children, "propertiesList")),
                )
            )
        else:
            log.debug(
                "While parsing groups this element ID had no type:'"
                + get_text_of_a_child(children, "id")
                + "'. Ignoring this element."
            )
    return result


def parsePropertiesList(propertiesList: ET.Element) -> list[GroupProperty]:
    result: list[GroupProperty] = []
    for children in propertiesList:
        if children.find("name") is not None:
            result.append(
                GroupProperty(
                    name=get_text_of_a_child(children, "name"),
                    id=get_text_of_a_child(children, "id"),
                    description=get_text_of_a_child(children, "description"),
                )
            )
    return result


def get_user_instruments(path: Path) -> list[UserInstrument]:
    result: list[UserInstrument] = []
    file_path: Path
    for file_path in path.glob("**/*.xml"):
        result.append(get_user_instrument(file_path))
    return result


@LogFileOpening
def get_user_instrument(file_path: Path) -> UserInstrument:
    tree_root: ET.Element = ET.parse(file_path).getroot()
    params = dictionary_from_list_in_xml_node(tree_root)

    return UserInstrument(
        name=file_path.name.replace(".xml", ""),
        # population=params["instrumentPop"],
        id=get_text_of_a_child(tree_root, "id"),
        params=params,
    )


def get_default_instruments(path: Path) -> list[DefaultInstrument]:
    result: list[DefaultInstrument] = []
    for i in path.glob("*.xml"):
        x = get_user_instrument(i)
        result.append(DefaultInstrument(name=x.name, params=x.params))
    return result


def get_text_of_a_child(element: ET.Element, tag: str) -> str:
    # This function mainly exists for the type checker
    x: ET.Element | None = element.find(tag)
    assert x is not None
    assert x.text is not None
    y: str = x.text
    return y


def get_child(element: ET.Element, tag: str) -> ET.Element:
    # This function mainly exists for the type checker
    x: ET.Element | None = element.find(tag)
    assert x is not None
    return x


def dictionary_from_list_in_xml_node(
    node: ET.Element,
) -> dict[str, str | int | float | bool]:
    # input node is either list or has a child named list

    iterable = None
    if node.tag == "list":
        iterable = node
    else:
        iterable = node.find("list")

    if iterable == None:
        raise RuntimeError("Input node doesn't have a child named list, nor is itself a list")

    result: dict[str, str | int | float | bool] = {}
    for el in iterable:
        key = get_text_of_a_child(el, "keyName")
        value = None
        match get_text_of_a_child(el, "typeAString"):
            case "float":
                value = float(get_text_of_a_child(el, "valueAsFloat"))
            case "long":
                value = int(get_text_of_a_child(el, "valueAsLong"))
            case "double":
                value = float(get_text_of_a_child(el, "valueAsDouble"))
            case "int":
                value = int(get_text_of_a_child(el, "valueAsInt"))
            case "bool":
                value = get_text_of_a_child(el, "valueAsBoolean") == "true"  # TODO check if correct
                print("CHECK IF FOLLOWING 2 LINES ARE EQUAL")
                print(get_text_of_a_child(el, "valueAsBoolean"))
                print(value)
            case "String":
                value = str(get_text_of_a_child(el, "valueAsString"))
            case _:
                raise RuntimeError("Undefined type in XML - " + get_text_of_a_child(el, "typeAString"))
        result[key] = value
    return result


def get_numerical_kernel_output(file_path: Path, instruments: list[UserInstrument]) -> NumericalResults:
    resulting_particle_detectors: list[ParticleDetector] = []
    for instrument in instruments:
        resulting_particle_detectors.append(get_particle_detector(file_path, instrument))

    return NumericalResults(
        emitted_currents=load_time_series(file_path / "emittedCurrents.txt"),
        number_of_superparticles=get_number_of_superparticles(file_path),
        particle_detectors=resulting_particle_detectors,
        time_steps=load_time_series(file_path / "Simulation_Control_-_time_steps_(s_._s)__TimeSteps.txt"),
        spis_log=(file_path / "SpisNum.log").read_text(encoding="utf_8", errors="backslashreplace"),
        total_current=load_time_series(file_path / "Total_current_on_spacecraft_surface._SCTotalCurrent.txt"),
        collected_currents=load_time_series(file_path / "collectedCurrents.txt"),
        surface_potential=load_time_series(
            file_path / "Average_surface_potential_of_node_0_(V_,_s)__ElecNode0_Potential.txt"
        ),
    )


def get_particle_detector(path: Path, instrument: UserInstrument) -> ParticleDetector:
    name = instrument.name
    assert isinstance(instrument.params["instrumentPop"], str)
    return ParticleDetector(
        name=name,
        population=instrument.params["instrumentPop"],
        differential_flux_2d=ordered_list_of_distribution2D(
            path, name + "_2D_DifferentialFlux_at_t=", "s.txt"
        ),
        differential_flux_mesh=ordered_list_of_meshes(path, name + "_3V_Differential_Flux_at_t=", "s.msh"),
        distribution_function_mesh=ordered_list_of_meshes(
            path, name + "_3V_Distribution_Function_at_t=", "s.msh"
        ),
        initial_distribution_mesh=ordered_list_of_meshes(
            path, name + "_3V_Initial_Distribution_Function_at_t=", "s.msh"
        ),
        angular2d_differential_flux=ordered_list_of_distribution2D(
            path, name + "_Angular2D_DifferentialFlux_at_t=", "s.txt"
        ),
        angular2d_function=ordered_list_of_distribution2D(path, name + "_Angular2DF_at_t=", "s.txt"),
        computationalOctree=ordered_list_of_meshes(path, name + "_computationalOctree_Time", "s.msh"),
        differential_flux_and_energy_df=ordered_list_of_distribution1D(
            path, name + "_Differential_Flux_and_Energy_DF_at_t=", "s.txt"
        ),
        initial_angular2df=ordered_list_of_distribution2D(path, name + "_Initial_Angular2DF_at_t=", "s.txt"),
        initial_velocity_2df=ordered_list_of_distribution2D(
            path, name + "_Initial_Velocity2DF_at_t=", "s.txt"
        ),
        moment=ordered_list_of_Moments(path, name + "_Moment_at_", "s.txt"),
        particle_list=ordered_list_of_particleLists(path, name + "_Particle_List_at_", "s.txt"),
        velocity_2df=ordered_list_of_distribution2D(path, name + "_Velocity2DF_at_t=", "s.txt"),
    )


def get_files_matching_start_and_end(path: Path, start: str, end: str) -> typing.Iterable[Path]:
    return path.glob(start + "*" + end)


def ordered_list_of_meshes(path: Path, start_of_file_name: str, end_of_file_name: str) -> list[Mesh]:
    result: list[Mesh] = []
    for i in get_files_matching_start_and_end(path, start_of_file_name, end_of_file_name):
        mesh = load_mesh(i)
        mesh.time = float(i.name.replace(start_of_file_name, "").replace(end_of_file_name, ""))
        result.append(mesh)

    result.sort(key=lambda mesh: (mesh.time is None, mesh.time))
    return result


@LogFileOpening
def load_mesh(path: Path) -> Mesh:
    """Loads mesh from path, does not guarantee that time will not be None"""
    mesh = meshio.read(path, file_format="gmsh")
    properties: list[str] = [x for x in mesh.cell_data.keys()]
    return Mesh(time=None, mesh=pyvista.wrap(mesh), properties=properties)


def ordered_list_of_distribution2D(
    path: Path, start_of_file_name: str, end_of_file_name: str
) -> list[Distribution2D]:
    result: list[Distribution2D] = []
    for i in get_files_matching_start_and_end(path, start_of_file_name, end_of_file_name):
        distribution = load_distribution2d(i)
        distribution.time = float(i.name.replace(start_of_file_name, "").replace(end_of_file_name, ""))
        result.append(distribution)

    result.sort(key=lambda distribution: (distribution.time is None, distribution.time))
    return result


def ordered_list_of_Moments(path: Path, start_of_file_name: str, end_of_file_name: str) -> list[Moments]:
    result: list[Moments] = []
    for i in get_files_matching_start_and_end(path, start_of_file_name, end_of_file_name):
        moments = load_moments(i)
        moments.time = float(i.name.replace(start_of_file_name, "").replace(end_of_file_name, ""))
        result.append(moments)

    result.sort(key=lambda moments: (moments.time is None, moments.time))
    return result


@LogFileOpening
def load_moments(path: Path) -> Moments:
    def string_to_vec(string: str) -> list[float]:
        return [float(i) for i in string.split(", ")]

    result: dict[str, float | list[float]] = {}

    data: list[str] = []
    with open(path, "r") as file:
        data = file.readlines()

    result["Moment of the distribution function: Density"] = float(data[2])
    result["Moment of the distribution function: Velocity in GMSH frame"] = string_to_vec(data[4])
    result["Moment of the distribution function: Mean energy"] = float(data[6])

    result["Moment of the flux distribution function at the detector surface : Flux"] = float(
        data[10].replace("m-1.s-2 ", "")
    )
    result[
        "Moment of the flux distribution function at the detector surface: Velocity in GMSH frame"
    ] = string_to_vec(data[12])
    result["Moment of the flux distribution function at the detector surface: Mean energy"] = float(data[14])

    result["Moment of the initial distribution function: Density"] = float(data[18])
    result["Moment of the initial distribution function: Velocity in GMSH frame"] = string_to_vec(data[20])
    result["Moment of the initial distribution function: Mean energy"] = float(data[22])

    return Moments(time=None, data=result)


def ordered_list_of_distribution1D(
    path: Path, start_of_file_name: str, end_of_file_name: str
) -> list[Distribution1D]:
    result: list[Distribution1D] = []
    for i in get_files_matching_start_and_end(path, start_of_file_name, end_of_file_name):
        distribution = load_distribution1d(i)
        distribution.time = float(i.name.replace(start_of_file_name, "").replace(end_of_file_name, ""))
        result.append(distribution)

    result.sort(key=lambda distribution: (distribution.time is None, distribution.time))
    return result


@LogFileOpening
def load_distribution1d(path: Path) -> Distribution1D:
    data: pandas.DataFrame = pandas.read_csv(path, sep="\t| ", engine="python")
    return Distribution1D(data=data, time=None)


@LogFileOpening
def load_time_series(path: Path) -> TimeSeries:
    data: pandas.DataFrame = pandas.read_csv(path, sep=", ", engine="python")
    return TimeSeries(data=data)


@LogFileOpening
def load_distribution2d(path: Path) -> Distribution2D:
    # print("reading file", path)

    # first we create xarray coords, from the first 3 lists of numbers

    # first we find interesting lines (named)
    lines_with_text: list[int] = []
    with open(path, "r") as file:
        for line_num, line in enumerate(file):
            if "[" in line:
                lines_with_text.append(line_num)

    x_size: int = lines_with_text[1] - lines_with_text[0] - 1
    y_size: int = lines_with_text[2] - lines_with_text[1] - 1
    z_size: int = lines_with_text[3] - lines_with_text[2] - 1
    # print(x_size, y_size, z_size)

    coords_dic: dict[str, list[float]] = {}
    dims: tuple[str, str, str]
    plotted_function: str

    x_cords: list[float] = []
    y_cords: list[float] = []
    z_cords: list[float] = []

    data: numpy.typing.ArrayLike

    data = np.zeros((x_size, y_size, z_size))

    with open(path, "r") as file:
        first_cord: str = file.readline().strip().strip("[]")
        for _ in range(x_size):
            x_cords.append(float(file.readline()))
        second_cord: str = file.readline().strip().strip("[]")
        for _ in range(y_size):
            y_cords.append(float(file.readline()))
        third_cord: str = file.readline().strip().strip("[]")
        for _ in range(z_size):
            z_cords.append(float(file.readline()))
        coords_dic[first_cord] = x_cords
        coords_dic[second_cord] = y_cords
        coords_dic[third_cord] = z_cords
        dims = (first_cord, second_cord, third_cord)
        plotted_function = file.readline().strip().strip("[]").strip()

        for z in range(z_size):
            for y in range(y_size):
                x = np.fromstring(str(file.readline()).strip(), dtype=float, sep=" ")
                # print(x[:])
                data[:, y, z] = x[:]
            file.readline()

    if np.count_nonzero(data) / data.size <= 0.1:  # if data is mostly empty we convert it to a sparse matrix
        data = sparse.COO.from_numpy(data)  # type:ignore

    result = xarray.DataArray(data=data, dims=dims, coords=coords_dic)
    return Distribution2D(time=None, data=result, plotted_function=plotted_function)


def ordered_list_of_particleLists(
    path: Path, start_of_file_name: str, end_of_file_name: str
) -> list[ParticleList]:
    result: list[ParticleList] = []
    for i in get_files_matching_start_and_end(path, start_of_file_name, end_of_file_name):
        particleLists = load_particle_list(i)
        particleLists.time = float(i.name.replace(start_of_file_name, "").replace(end_of_file_name, ""))
        result.append(particleLists)

    result.sort(key=lambda particleLists: (particleLists.time is None, particleLists.time))
    return result


@LogFileOpening
def load_particle_list(path: Path) -> ParticleList:
    with open(path, "r") as file:
        x = file.readline()
        y = file.readline()
        z = file.readline()

    names = z.strip().strip("#").strip().split(" ")
    info = x + y + z

    data: pandas.DataFrame = pandas.read_csv(
        path, sep="\t| ", engine="python", header=None, skiprows=[0, 1, 2], names=names
    )
    return ParticleList(data=data, time=None, info=info)


def get_number_of_superparticles(path: Path) -> list[NumberOfSuperparticles]:
    result: list[NumberOfSuperparticles] = []
    for i in get_files_matching_start_and_end(path, "Number_of_", "SPNB.txt"):
        result.append(load_number_of_superparticles(i))
    return result


def load_number_of_superparticles(path: Path) -> NumberOfSuperparticles:
    population: str = path.name.replace("Number_of_", "")
    population = population[: population.find("(_._s)")].strip().strip("_")
    data = load_time_series(path)
    return NumberOfSuperparticles(data=data, population=population)


def load_from_SPIS(path: Path) -> Simulation:
    log.info("Started loading SPIS simulation at " + str(path))

    data = load_data(path)
    log.info("Done loading SPIS simulation at" + str(path))
    return data


def save_as_pickle(simulation: Simulation, path: Path):
    # Serialize the object
    serialized_data = pickle.dumps(simulation)
    log.info(
        "The data has the size of approximately "
        + str(len(pickle.dumps(serialized_data)) // 1000**2)
        + " megabytes"
    )
    with open(path, "wb") as f:
        f.write(serialized_data)
    log.info("Saved the data to " + str(path))


@LogFileOpening
def load_pickle(path: Path) -> Simulation:
    with open(path, "rb") as f:
        deserialized_object = pickle.load(f)
    assert isinstance(deserialized_object, Simulation)
    return deserialized_object


def get_extracted_datafields(path: Path) -> ExtractedDataFields:
    spacecraft_face = load_mesh(path / "Spacecraft_FACE.msh")
    spacecraft_vertex = load_mesh(path / "Spacecraft_VERTEX.msh")
    volume_vertex = load_mesh(path / "Volume_VERTEX.msh")

    spacecraft_mesh_file = list((path / "../../../../Preprocessing/Mesh/GeometricalSystem").glob("*.msh"))[0]
    spacecraft_mesh_name = spacecraft_mesh_file.name
    spacecraft_mesh = load_mesh(spacecraft_mesh_file)

    display_vol_mesh_file = list(path.glob("DisplayVolMesh*.msh"))[0]
    display_vol_mesh_name = display_vol_mesh_file.name
    display_vol_mesh = load_mesh(display_vol_mesh_file)

    all_datasets: list[Path] = []
    time_series: list[Path] = []  # We are not doing anything with this, for now

    for i in path.glob("*.nc"):
        # Following lines filter out time series and masks
        if "time" in i.name:
            time_series.append(i)
            continue
        if any(banned_str in i.name for banned_str in ["VERTEX", "POLYHEDRON", "FACE"]):
            continue
        all_datasets.append(i)

    for i in all_datasets:
        data: xarray.Dataset = xarray.open_dataset(i)
        mask: xarray.Dataset = xarray.open_dataset(i / ".." / data.attrs["meshMaskURI"])
        mesh: Mesh

        if mask.attrs["meshURI"] == "Spacecraft_FACE.msh":
            mesh = spacecraft_face
        elif mask.attrs["meshURI"] == "Spacecraft_VERTEX.msh":
            mesh = spacecraft_vertex
        elif mask.attrs["meshURI"] == "Volume_VERTEX.msh":
            mesh = volume_vertex
        elif mask.attrs["meshURI"] == (
            "../../../../Preprocessing/Mesh/GeometricalSystem/" + str(spacecraft_mesh_name)
        ):
            mesh = spacecraft_mesh
        elif display_vol_mesh_name == mask.attrs["meshURI"]:
            mesh = display_vol_mesh
        else:  # If the mesh isnt one the 5 types we skip the data
            log.error("Trying to add data to an unknown mesh, skipping")
            log.error("The mesh name is " + str(mask.attrs["meshURI"]))
            continue

        if not check_mask_is_identity(mask, data.attrs["meshMaskURI"]):
            uri = str(data.attrs["meshMaskURI"])
            log.debug(f"Mask {uri} was not an identity")
            temp = reshape_data_according_to_mask(data, mask)

            add_data_to_mesh(da=temp, mesh=mesh, path_to_property=i, mask=mask)
            continue

        # https://stackoverflow.com/questions/74693202/add-point-data-to-mesh-and-save-as-vtu
        for _, da in data.data_vars.items():  # type:ignore
            cur_data_array: xarray.DataArray = da  # type:ignore
            add_data_to_mesh(da=cur_data_array, mesh=mesh, path_to_property=i, mask=mask)  # type:ignore
            continue

    return ExtractedDataFields(
        spacecraft_face=spacecraft_face,
        spacecraft_vertex=spacecraft_vertex,
        volume_vertex=volume_vertex,
        spacecraft_mesh=spacecraft_mesh,
        display_vol_mesh=display_vol_mesh,
    )


def add_data_to_mesh(da: xarray.DataArray, mesh: Mesh, path_to_property: Path, mask: xarray.Dataset):
    try:
        cur_data = da.data
        if len(cur_data) == mesh.mesh.number_of_points:
            mesh.mesh.point_data[path_to_property.stem] = cur_data
        elif len(cur_data) == mesh.mesh.number_of_cells:
            mesh.mesh.cell_data[path_to_property.stem] = cur_data
        # this last option shouldn't run
        else:
            log.warn(f"{str(path_to_property.stem)} was saved as field data. This data can't be plotted")
            mesh.mesh.field_data[path_to_property.stem] = cur_data
        mesh.properties.append(path_to_property.stem)
        log.debug("Loaded " + path_to_property.stem)
    except Exception as e:
        log.warn("Failed on " + path_to_property.stem + " this data won't be available")
        log.debug("Was using the mask " + str(mask.attrs["meshURI"]))
        log.debug(type(e).__name__, e)


def check_mask_is_identity(dataset: xarray.Dataset, mask_name: str) -> bool:
    # Could be made more rigorous
    x: xarray.DataArray = dataset["meshElmentId"]
    mask_as_series: pandas.Series[int] = x.to_series()  # type:ignore

    return mask_as_series.is_monotonic_increasing


def reshape_data_according_to_mask(data: xarray.Dataset, mask: xarray.Dataset) -> xarray.DataArray:
    mask_as_series: list[int] = mask["meshElmentId"].to_series().to_list()  # type:ignore

    data_array: xarray.DataArray | None = None
    for _, j in data.data_vars.items():
        data_array = j

    assert data_array is not None

    if abs(data_array.min() - data_array.max()) < np.finfo(float).eps * 4:
        uri = str(data.attrs["meshMaskURI"])
        log.debug(f"Not permuting according to {uri} because it's identically almost zero.")
        return data_array

    if data.attrs["meshMaskURI"] == "surfFlagSC-FACEMask.nc":  # TODO
        uri = str(data.attrs["meshMaskURI"])
        log.error(f"Not permuting according to {uri}, not implemented yet.")
        return data_array

    np_data_array: numpy.typing.NDArray[np.float_] = data_array.to_numpy()  # type:ignore

    assert np_data_array is not None

    inverted_indices = [0] * len(mask_as_series)

    # Invert the indices
    for i, idx in enumerate(mask_as_series):
        inverted_indices[idx] = i

    mask_as_series = inverted_indices

    result = [np_data_array[i] for i in mask_as_series]
    result = np.array(result)

    return xarray.DataArray(result)


def load_simulation(
    path: Path,
    *,
    processed_name: str = "processed_simulation.pkl",
    force_processing: bool = False,
) -> Simulation:
    if force_processing or not (path / processed_name).exists():
        save_as_pickle(load_from_SPIS(path), path / processed_name)

    result = load_pickle(path / processed_name)

    return result
