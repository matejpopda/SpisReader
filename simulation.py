import logging
from dataclasses import dataclass, field
import pyvista
import pyvista.core.pointset
import pandas
import xarray
import typing
import weakref
import pathlib


log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DefaultInstrument:
    """Class encapsulating a single instrument

    Attributes:
        name (str): name of the instrument
        params (dict): parameters of the instrument
    """

    name: str
    params: dict[str, str | int | float | bool]


@dataclass(kw_only=True)
class UserInstrument:
    """Class encapsulating a single user instrument

    Attributes:
        name (str): name of the user instrument
        id (str): id of the user instrument
        params (dict): paramaters of the user instrument
    """

    name: str
    id: str
    params: dict[str, str | int | float | bool]


@dataclass(kw_only=True)
class Group:
    """Class encapsulating a single physical group

    Attributes:
        name (str): name of the group
        SPISid (str): id of the group in SPIS
        GMSHid (str): id of the group in GMSH (should be the physical group)
        type (str): type of the group
        properties (list[GroupProperty]): properties of the group
    """

    name: str
    SPISid: str
    GMSHid: str
    type: str
    properties: list["GroupProperty"]


@dataclass(kw_only=True)
class GroupProperty:
    """Class encapsulating group properties

    Attributes:
        name (str): name of the property
        id (str): id of the property
        description(str): description of the property
    """

    name: str
    id: str
    description: str


@dataclass(kw_only=True)
class SimulationPreprocessing:
    """Class encapsulating data from the preprocessing folder

    Attributes:
        model (Mesh): Contains the mesh from Preprocessing/Mesh/GeometricalSystem/*.msh, this file contains the base model of the spacecraft
        groups (list[Group]): Contains data from Preprocessing/Groups/groups.xml, this file contains information about declared groups in the simulation
        electrical_circuit (str): Contains the copy of the Preprocessing/ElectricalCircuit/circuit.txt file
    """

    model: "Mesh"
    groups: list[Group]
    electrical_circuit: str


@dataclass(kw_only=True)
class SimulationResults:
    """Results of a simulation (and variables that change per simulation, such as instruments are stored here)

    Attributes:
        default_instruments (list[DefaultInstrument]): Contains all default instruments, e.g. /Simulations/Run1/DefaultInstruments
        user_instruments (list[UserInstrument]): Contains all user instruments, e.g. /Simulations/Run1/UserInstruments
        global_parameters (dict[str, str|int|float|bool]): Contains global parameters, e.g. /Simulations/Run1/Global Parameters
        numerical_kernel_output (NumericalResults): Contains output from the numerical kernel, e.g. /Simulations/Run1/NumKernel/Output
        monitored_data_fields (None): Is empty for now, in the future it might contain  /Simulations/Run1/OutputFolder/DataFieldMonitored
        extracted_data_fields (ExtractedDataFields): Contains extracted datafields, e.g. /Simulations/Run1/OutputFolder/DataFieldExtracted

    """

    default_instruments: list[DefaultInstrument]
    user_instruments: list[UserInstrument]
    global_parameters: dict[str, str | int | float | bool]
    numerical_kernel_output: "NumericalResults"
    monitored_data_fields: None
    extracted_data_fields: "ExtractedDataFields"


@dataclass(kw_only=True)
class ExtractedDataFields:
    """Contains all the mesh information obtained from the extracted data fields folder on 4 different meshes"""

    spacecraft_vertex: "Mesh"
    spacecraft_face: "Mesh"
    volume_vertex: "Mesh"
    spacecraft_mesh: "Mesh"
    display_vol_mesh: "Mesh"


    particle_trajectories: list["ParticleTrajectory"]
    """*trajectory.nc_mesh.msh"""

@dataclass(kw_only=True)
class ParticleTrajectory:
    name: str
    particle_id: int 
    time: float

    data: list[tuple[float, float,float]]



@dataclass(kw_only=True)
class NumericalResults:
    """Encapsulates data from the numerical kernel"""

    surface_potential: "TimeSeries"
    """Average_surface_potential_of_node"""
    collected_currents: "TimeSeries"
    """collectedCurrents.txt"""

    emitted_currents: "TimeSeries"
    """emittedCurrents.txt"""

    number_of_superparticles: list["NumberOfSuperparticles"]
    """Number_of_*.txt"""

    particle_detectors: list["ParticleDetector"]
    """ParticleDetector[number]_[population]_*"""

    time_steps: "TimeSeries"
    """Simulation_Control_-_time_steps*.txt"""

    spis_log: str
    """SpisNum.log"""

    total_current: "TimeSeries"
    """Total_current_on_spacecraft_surface*.txt"""





@dataclass(kw_only=True)
class NumberOfSuperparticles:
    population: str
    data: "TimeSeries"


@dataclass(kw_only=True)
class ParticleDetector:
    """Class encapsulating a particle detector"""

    name: str
    """Name of the particle detector"""
    population: str
    """The type of population the particle detector measures"""

    differential_flux_2d: list["Distribution2D"]
    """[name]_2D_DifferentialFlux_at_t=*s.txt"""

    differential_flux_mesh: list["Mesh"]
    """[name]_3V_Differential_Flux_at_t=*s.msh"""

    distribution_function_mesh: list["Mesh"]
    """[name]_3V_Distribution_Function_at_t=*s.msh"""

    initial_distribution_mesh: list["Mesh"]
    """[name]_3V_Initial_Distribution_Function_at_t=*s.msh"""

    angular2d_differential_flux: list["Distribution2D"]
    """[name]_Angular2D_DifferentialFlux_at_t=*s.txt"""

    angular2d_function: list["Distribution2D"]
    """[name]_Angular2DF_at_t=*s.txt"""

    computationalOctree: list["Mesh"]
    """[name]_computationalOctree_Time*s.msh"""

    differential_flux_and_energy_df: list["Distribution1D"]
    """[name]_Differential_Flux_and_Energy_DF_at_t=*s.txt"""

    initial_angular2df: list["Distribution2D"]
    """[name]_Initial_Angular2DF_at_t=*s.txt"""

    initial_velocity_2df: list["Distribution2D"]
    """[name]_Initial_Velocity2DF_at_t=*s.txt"""

    moment: list["Moments"]
    """[name]_Moment_at_*s.txt"""

    particle_list: list["ParticleList"]
    """[name]_Particle_List_at_*s.txt"""

    velocity_2df: list["Distribution2D"]
    """[name]_Velocity2DF_at_t=*s.txt"""


    def __str__(self):
        return "Particle detector - " + self.name


@dataclass(kw_only=True, weakref_slot=True, slots=True, unsafe_hash=True)
class Mesh:
    """Class encapsulating a plottable mesh with functions from plotters.py"""

    name: str = field(hash=True)
    mesh: pyvista.core.pointset.UnstructuredGrid = field(hash=False)
    time: float | None
    """In case there is multiple meshes made at different times, this parameter can be used to for example sort them"""
    properties: list[str] = field(hash=False)
    """List of plottable properties on the mesh"""

    loadable_properties: dict[str, pathlib.Path] = field(hash=False)

    instance_list: typing.ClassVar[weakref.WeakSet["Mesh"]] = weakref.WeakSet()

    def __post_init__(self):
        self.__class__.instance_list.add(self)


    def __str__(self):
        return "Mesh class - " + self.name


@dataclass(kw_only=True)
class TimeSeries:
    """Simple class encapsulating a time series"""

    data: pandas.DataFrame


@dataclass(kw_only=True, weakref_slot=True, slots=True, unsafe_hash=True)
class Distribution2D:
    """Class encapsulating 2D distribution"""

    time: float | None
    """At what time was the distribution saved"""
    data: xarray.DataArray | None = field(hash=False)
    """DataArray with named coordinates, the array can sometimes be sparse, if it is None then it is not loaded"""
    plotted_function: str
    """What function is being plotted"""

    path_to_data: pathlib.Path

    instance_list: typing.ClassVar[weakref.WeakSet["Distribution2D"]] = weakref.WeakSet()

    def __post_init__(self):
        self.__class__.instance_list.add(self)




@dataclass(kw_only=True)
class Distribution1D:
    """Class encapsulating 1D distribution"""

    time: float | None
    """At what time was the distribution saved"""
    data: pandas.DataFrame


@dataclass(kw_only=True)
class Moments:
    """Class encapsulating the moments file, contains some information about the moments"""

    time: float | None
    """At what time was this output saved"""
    data: dict[str, float | list[float]]


@dataclass(kw_only=True, weakref_slot=True, slots=True, unsafe_hash=True)
class ParticleList:
    """Class encapsulating particle lists"""

    time: float | None
    data: pandas.DataFrame | None = field(hash=False)
    info: str
    path: pathlib.Path

    instance_list: typing.ClassVar[weakref.WeakSet["ParticleList"]] = weakref.WeakSet()

    def __post_init__(self):
        self.__class__.instance_list.add(self)



@dataclass(kw_only=True)
class Simulation:
    """Class encapsulating the whole simulation output from SPIS"""

    preprocessing: SimulationPreprocessing
    results: SimulationResults

    @property
    def extracted_data_fields(self):
        return self.results.extracted_data_fields
    

    @property
    def user_instruments(self):
        return self.results.user_instruments

    @property
    def particle_detectors(self):
        return self.results.numerical_kernel_output.particle_detectors

    @property 
    def time_steps(self):
        return self.results.numerical_kernel_output.time_steps