import logging
from dataclasses import dataclass
import pyvista
import pyvista.core.dataset
import pandas
import xarray
# from spisModule.reader import get_user_instruments, get_default_instruments, get_extracted_datafields, get_numerical_kernel_output,get_groups, load_mesh, dictionary_from_list_in_xml_node

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DefaultInstrument:
    """Class encapsulating a single instrument"""
    name :str
    params: dict[str, str|int|float|bool]

@dataclass(kw_only=True)
class UserInstrument:
    """Class encapsulating a single instrument"""
    name :str
    id: str
    params: dict[str, str|int|float|bool]

@dataclass(kw_only=True)
class Group:
    """Class encapsulating a single physical group"""
    name:str 
    SPISid:str 
    GMSHid:str 
    type:str
    properties:list["GroupProperty"]

@dataclass(kw_only=True)
class GroupProperty:
    """Class encapsulating group properties"""
    name:str
    id:str
    description:str

@dataclass(kw_only=True)
class SimulationPreprocessing:
    """Data from preprocessing is stored here"""
    model: "Mesh"
    groups: list[Group]
    electrical_circuit : str

    # def __init__(self, path_to_preprocessing: Path) -> None:
    #     model_path : list[Path] = list((path_to_preprocessing / "Mesh" / "GeometricalSystem").glob("*"))
    #     assert len(model_path) == 1
    #     self.model: Mesh = load_mesh(model_path[0]) 
    #     # Loads file Preprocessing/Mesh/GeometricalSystem/*.msh 
    #     # Probably can be the same class

    #     self.groups: list[Group] = get_groups(path_to_preprocessing / "Groups" / "groups.xml") 
    #     # \Preprocessing\Groups/groups.xml

    #     self.electrical_circuit : str = (path_to_preprocessing / "ElectricalCircuit" / "circuit.txt").read_text() 
    #     # \Preprocessing\ElectricalCircuit\circuit.txt


@dataclass(kw_only=True)
class SimulationResults:
    """Results of a simulation (and variables that change per simulation, such as instruments are stored here)"""
    default_instruments : list[DefaultInstrument]
    user_instruments : list[UserInstrument]
    global_parameters : dict[str, str|int|float|bool]
    numerical_kernel_output : "NumericalResults"
    monitored_data_fields : None
    extracted_data_fields : "ExtractedDataFields"

    # def __init__(self, path_to_results: Path) -> None:
    #     self.default_instruments : list[DefaultInstrument] = get_default_instruments(path_to_results / "DefaultInstruments")
    #     # List of instruments from \Simulations\Run1\DefaultInstruments

    #     self.user_instruments : list[UserInstrument] = get_user_instruments(path_to_results / "UserInstruments") 
    #     # List of instruments from \Simulations\Run1\UserInstruments


    #     self.global_parameters = dictionary_from_list_in_xml_node(ET.parse(path_to_results / "GlobalParameters" / "globalParameters.xml").getroot()) 
    #     # \Simulations\Run1\GlobalParameters

    #     self.numerical_kernel_output : NumericalResults = get_numerical_kernel_output(path_to_results / "NumKernel" / "Output", self.user_instruments) 
    #     # \Simulations\Run1\NumKernel\Output

    #     self.monitored_data_fields = None # TODO I think the data can be more easily gotten from numerical kernel


    #     self.extracted_data_fields:ExtractedDataFields = get_extracted_datafields(path_to_results / "OutputFolder" / "DataFieldExtracted")
    #     # \Simulations\Run1\OutputFolder\DataFieldExtracted




@dataclass(kw_only=True)
class ExtractedDataFields: 
    spacecraft_vertex : "Mesh"
    spacecraft_face : "Mesh"
    volume_vertex : "Mesh"
    spacecraft_mesh: "Mesh"


@dataclass(kw_only=True)
class NumericalResults:
    surface_potential: "TimeSeries"
    # \Average_surface_potential_of_nod
    collected_currents: "TimeSeries"
    # \collectedCurrents.txt

    emitted_currents: "TimeSeries"
    # \emittedCurrents.txt

    number_of_superparticles :list["NumberOfSuperparticles"]
    #  \Number_of_*.txt

    particle_detectors : list["ParticleDetector"]
    # \ParticleDetector[number]_[population]_*

    time_steps: "TimeSeries"
    # \Simulation_Control_-_time_steps*.txt
    
    spis_log :str
    # \SpisNum.log
    
    total_current: "TimeSeries"
    # \Total_current_on_spacecraft_surface*.txt

@dataclass(kw_only=True)
class NumberOfSuperparticles:
    population:str
    data:"TimeSeries"

@dataclass(kw_only=True)
class ParticleDetector:
    name :str
    population :str

    differential_flux_2d :list["Distribution2D"]
    #\[name]_2D_DifferentialFlux_at_t=*s.txt

    differential_flux_mesh :list["Mesh"]
    #\[name]_3V_Differential_Flux_at_t=*s.msh

    distribution_function_mesh :list["Mesh"]
    #\[name]_3V_Distribution_Function_at_t=*s.msh

    initial_distribution_mesh :list["Mesh"]
    #\[name]_3V_Initial_Distribution_Function_at_t=*s.msh

    angular2d_differential_flux :list["Distribution2D"]
    #\[name]_Angular2D_DifferentialFlux_at_t=*s.txt

    angular2d_function :list["Distribution2D"]
    #\[name]_Angular2DF_at_t=*s.txt

    computationalOctree : list["Mesh"]
    #\[name]_computationalOctree_Time*s.msh

    differential_flux_and_energy_df : list["Distribution1D"]
    #\[name]_Differential_Flux_and_Energy_DF_at_t=*s.txt

    initial_angular2df : list["Distribution2D"]
    #\[name]_Initial_Angular2DF_at_t=*s.txt

    initial_velocity_2df : list["Distribution2D"]
    #\[name]_Initial_Velocity2DF_at_t=*s.txt

    moment : list["Moments"]
    #\[name]_Moment_at_*s.txt

    particle_list : list["ParticleList"]
    #\[name]_Particle_List_at_*s.txt

    velocity_2df : list["Distribution2D"] 
    #\[name]_Velocity2DF_at_t=*s.txt


@dataclass(kw_only=True)
class Mesh:
    mesh:pyvista.core.pointset.UnstructuredGrid
    time: float|None
    properties: list[str]

@dataclass(kw_only=True)
class TimeSeries:
    data: pandas.DataFrame

@dataclass(kw_only=True)
class Distribution2D:
    time: float|None
    data : xarray.DataArray
    plotted_function : str

@dataclass(kw_only=True)
class Distribution1D:
    time: float|None
    data: pandas.DataFrame

@dataclass(kw_only=True)
class Moments:
    time: float|None
    data : dict[str, float|list[float]]

@dataclass(kw_only=True)
class ParticleList:
    time: float|None
    data: pandas.DataFrame
    info: str

@dataclass(kw_only=True)
class Simulation:
    """Wrapper for preprocessing information and results"""
    preprocessing : SimulationPreprocessing
    results : SimulationResults

    @property
    def extracted_data_fields(self):
        return self.results.extracted_data_fields
    
    # def get_data_field_property(self, property: str) -> Mesh:
        
