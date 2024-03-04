import typing
from pathlib import Path
import xml.etree.ElementTree as ET
import logging as log
from dataclasses import dataclass



class DefaultInstrument:
    """Class encapsulating a single instrument"""
    def __init__(self) -> None:
        pass

@dataclass(kw_only=True)
class UserInstrument:
    """Class encapsulating a single instrument"""
    name :str
    id: str
    params: dict

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

class SimulationPreprocessing:
    """Data from preprocessing is stored here"""
    def __init__(self, path_to_preprocessing: Path) -> None:
        self.model: Mesh = None 
        # Loads file Preprocessing/Mesh/GeometricalSystem/model.msh 
        # Probably can be the same class

        self.groups: list[Group] = get_groups(path_to_preprocessing / "Groups" / "groups.xml") 
        # \Preprocessing\Groups/groups.xml

        self.electrical_circuit : str = (path_to_preprocessing / "ElectricalCircuit" / "circuit.txt").read_text() 
        # \Preprocessing\ElectricalCircuit\circuit.txt


class SimulationResults:
    """Results of a simulation (and variables that change per simulation, such as instruments are stored here)"""
    def __init__(self, path_to_results: Path) -> None:
        self.default_instruments : list[DefaultInstrument] = None #get_default_instruments(path_to_results / "DefaultInstruments")
        # List of instruments from \Simulations\Run1\DefaultInstruments

        self.user_instruments : list[UserInstrument] = get_user_instruments(path_to_results / "UserInstruments") 
        # List of instruments from \Simulations\Run1\UserInstruments

        self.global_parameters = dictionary_from_list_in_xml_node(ET.parse(path_to_results / "GlobalParameters" / "globalParameters.xml").getroot()) 
        # \Simulations\Run1\GlobalParameters

        self.numerical_kernel_output : list[NumericalResults] = None 
        # \Simulations\Run1\NumKernel\Output

class NumericalResults:
    def __init__(self) -> None:
        self.surface_potential: TimeSeries = None
        # \Average_surface_potential_of_node_0

        self.collected_currents: TimeSeries = None
        # \collectedCurrents.txt

        self.emitted_currents: TimeSeries = None
        # \emittedCurrents.txt

        self.number_of_superparticles :list[NumberOfSuperparticles] = None
        #  \Number_of_*.txt

        self.particle_detectors : list[ParticleDetector] = None
        # \ParticleDetector[number]_[population]_*

        self.time_steps: TimeSeries = None
        # \Simulation_Control_-_time_steps*.txt
        
        self.spis_log :str = None
        # \SpisNum.log
        
        self.total_current: TimeSeries = None
        # \Total_current_on_spacecraft_surface*.txt

class NumberOfSuperparticles:
    def __init__(self) -> None:
        self.population = None
        self.data = None


class ParticleDetector:
    def __init__(self) -> None:
        self.name :str = None
        self.population :str = None

        self.differential_flux_2d :list[Distribution2D] = None
        #\[name]_2D_DifferentialFlux_at_t=*s.txt

        self.differential_flux_mesh :list[Mesh] = None
        #\[name]_3V_Differential_Flux_at*.msh

        self.distribution_function_mesh :list[Mesh] = None
        #\[name]_3V_Distribution_Function_at*.msh

        self.initial_distribution_mesh :list[Mesh] = None
        #\[name]_3V_Initial_Distribution_Function_at*.msh

        self.angular2d_differential_flux :list[Distribution2D] = None
        #\[name]_Angular2D_DifferentialFlux_at_t=*s.txt

        self.angular2d_function :list[Distribution2D] = None
        #\[name]_Angular2DF_at_t=*s.txt

        self.computationalOctree : list[Mesh] = None
        #\[name]_computationalOctree_Time*.msh

        self.differential_flux_and_energy_df : list[Distribution1D] = None
        #\[name]_Differential_Flux_and_Energy_DF_at_t=*s.txt

        self.initial_angular2df : list[Distribution2D] = None
        #\[name]_Initial_Angular2DF_at_t=*s.txt

        self.initial_velocity_2df : list[Distribution2D] = None
        #\[name]_Initial_Velocity2DF_at_t=*s.txt

        self.moment : list[Moments] = None
        #\[name]_Moment_at_*s.txt

        self.particle_list : list[ParticleList] = None
        #\[name]_Particle_List_at_*s.txt

        self.velocity_2df : list[Distribution2D] = None 
        #\[name]_Velocity2DF_at_t=*s.txt



class Mesh:
    pass

class TimeSeries:
    pass

class Distribution2D:
    pass

class Distribution1D:
    pass

class Moments:
    pass

class ParticleList:
    #TODO maybe not needed
    pass


class Simulation:
    """Wrapper for preprocessing information and results"""
    def __init__(self, preprocessing, results) -> None:
        self.preprocessing : SimulationPreprocessing = preprocessing
        self.results : SimulationResults = results

def load_data(path: Path) -> Simulation:
    
    results = SimulationResults(path / "Simulations" / "Run1" )
    preprocessing = SimulationPreprocessing(path / "Preprocessing")

    return Simulation(preprocessing, results)

def get_default_instruments(path: Path) -> list[DefaultInstrument]:
    print(path.exists())
    for instrument in path.glob("*"):
        print(instrument)

def get_groups(path: Path) -> list[Group]:
    if not path.exists(): 
        raise FileNotFoundError("Missing file " +  str(path))
    
    result = []
    
    tree = ET.parse(path)
    groupList = tree.getroot()[0]
    for children in groupList:
        if children.find("type") is not None:
            result.append(Group(
                name = children.find("name").text,
                SPISid = children.find("id").text,
                GMSHid = children.find("linkedMeshMaskIds").find("int").text,
                type = children.find("type").text,
                properties = parsePropertiesList(children.find("propertiesList"))
            ))
        else: 
            log.debug("While parsing groups this element ID had no type:'" + children.find("id").text + "'. Ignoring this element." )
    return result


def parsePropertiesList(propertiesList: ET.Element) -> list[GroupProperty]:
    result = []
    for children in propertiesList:
        if children.find("name") is not None:
            result.append(GroupProperty(
                name = children.find("name").text,
                id = children.find("id").text,
                description = children.find("description").text
            ))
    return result

def get_user_instruments(path: Path) -> list[UserInstrument]:
    result = []
    file_path: Path
    for file_path in path.glob("**/*.xml"):
        result.append(get_user_instrument(file_path))
    return result

def get_user_instrument(file_path:Path) -> UserInstrument:
    tree_root :ET.Element = ET.parse(file_path).getroot()

    population=None

    params = dictionary_from_list_in_xml_node(tree_root)

    return UserInstrument(
        name=file_path.name,
        # population=params["instrumentPop"],
        id=tree_root.find("id").text,
        params=params,
    )

def dictionary_from_list_in_xml_node(node: ET.Element) -> dict:
    # input node is either list or has a child named list
    
    iterable = None
    if node.tag == "list":
        iterable = node
    else:
        iterable = node.find("list")

    if iterable == None:
        raise RuntimeError("Input node doesn't have a child named list, nor is itself a list")

    result: dict = {} 
    for el in iterable:
        key = el.find("keyName").text
        value = None
        match el.find("typeAString").text:
            case "int":
                value = el.find("valueAsInt").text
            case "float":
                value = el.find("valueAsFloat").text
            case "long":
                value = el.find("valueAsLong").text
            case "double":
                value = el.find("valueAsDouble").text
            case "int":
                value = el.find("valueAsInt").text
            case "bool":
                value = el.find("valueAsBoolean").text
            case "String":
                value = el.find("valueAsString").text
            case _:
                raise RuntimeError("Undefined type in XML - " + el.find("typeAString").text)
        result[key] = value
    return result


if __name__=="__main__":
    log.basicConfig(level=2)
    # groups = get_groups(Path("C:\\Users\\matej\\Desktop\\VU\\8\\DefaultProject.spis5\\DefaultStudy\\Preprocessing\\Groups\\groups.xml"))
    # user_instruments = get_user_instruments(Path("C:\\Users\\matej\\Desktop\\VU\\8\\DefaultProject.spis5\\DefaultStudy\\Simulations\\Run1\\UserInstruments"))
    # path = Path("C:/Users/matej/Desktop/VU/8/DefaultProject.spis5")  / "DefaultStudy"
    path = Path("C:/Users/matej/Desktop/VU/example/example/cube_wsc_01.spis5")  / "CS_01"
    data = load_data(path)
    pass