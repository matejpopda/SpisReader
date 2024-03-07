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
    params: dict[str, str]

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
        self.model: Mesh = load_mesh(path_to_preprocessing / "Mesh" / "GeometricalSystem" / "model.msh") 
        # Loads file Preprocessing/Mesh/GeometricalSystem/model.msh 
        # Probably can be the same class

        self.groups: list[Group] = get_groups(path_to_preprocessing / "Groups" / "groups.xml") 
        # \Preprocessing\Groups/groups.xml

        self.electrical_circuit : str = (path_to_preprocessing / "ElectricalCircuit" / "circuit.txt").read_text() 
        # \Preprocessing\ElectricalCircuit\circuit.txt


class SimulationResults:
    """Results of a simulation (and variables that change per simulation, such as instruments are stored here)"""
    def __init__(self, path_to_results: Path) -> None:
        self.default_instruments : list[DefaultInstrument] = get_default_instruments(path_to_results / "DefaultInstruments")
        # List of instruments from \Simulations\Run1\DefaultInstruments

        self.user_instruments : list[UserInstrument] = get_user_instruments(path_to_results / "UserInstruments") 
        # List of instruments from \Simulations\Run1\UserInstruments


        self.global_parameters = dictionary_from_list_in_xml_node(ET.parse(path_to_results / "GlobalParameters" / "globalParameters.xml").getroot()) 
        # \Simulations\Run1\GlobalParameters

        self.numerical_kernel_output : NumericalResults = get_numerical_kernel_output(path_to_results / "NumKernel" / "Output", self.user_instruments) 
        # \Simulations\Run1\NumKernel\Output

        self.monitored_data_fields = None

        self.extracted_data_fields = None 
        
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
    #\[name]_3V_Differential_Flux_at*.msh

    distribution_function_mesh :list["Mesh"]
    #\[name]_3V_Distribution_Function_at*.msh

    initial_distribution_mesh :list["Mesh"]
    #\[name]_3V_Initial_Distribution_Function_at*.msh

    angular2d_differential_flux :list["Distribution2D"]
    #\[name]_Angular2D_DifferentialFlux_at_t=*s.txt

    angular2d_function :list["Distribution2D"]
    #\[name]_Angular2DF_at_t=*s.txt

    computationalOctree : list["Mesh"]
    #\[name]_computationalOctree_Time*.msh

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


@dataclass(kw_only=True)
class Simulation:
    """Wrapper for preprocessing information and results"""
    preprocessing : SimulationPreprocessing
    results : SimulationResults

def load_data(path: Path) -> Simulation:
    return Simulation(
        results = SimulationResults(path / "Simulations" / "Run1" ),
        preprocessing = SimulationPreprocessing(path / "Preprocessing")
    )

def get_default_instruments(path: Path) -> list[DefaultInstrument]:
    # print(path.exists())
    # for instrument in path.glob("*"):
    #     print(instrument)
    return None  # type: ignore #TODO

def get_groups(path: Path) -> list[Group]:
    if not path.exists(): 
        raise FileNotFoundError("Missing file " +  str(path))
    
    result:list[Group] = []
    
    tree = ET.parse(path)
    groupList = tree.getroot()[0]
    for children in groupList:
        if children.find("type") is not None:

            result.append(Group(
                name = get_text_of_a_child(children, "name"),
                SPISid = get_text_of_a_child(children, "id"),
                GMSHid = get_text_of_a_child(get_child(children,"linkedMeshMaskIds"), "int"),
                type = get_text_of_a_child(children, "type"),
                properties = parsePropertiesList(get_child(children, "propertiesList"))
            ))
        else: 
            log.debug("While parsing groups this element ID had no type:'" + get_text_of_a_child(children, "id") + "'. Ignoring this element." )
    return result


def parsePropertiesList(propertiesList: ET.Element) -> list[GroupProperty]:
    result: list[GroupProperty] = []
    for children in propertiesList:
        if children.find("name") is not None:
            result.append(GroupProperty(
                name = get_text_of_a_child(children, "name"),
                id = get_text_of_a_child(children, "id"),
                description = get_text_of_a_child(children, "description")
            ))
    return result

def get_user_instruments(path: Path) -> list[UserInstrument]:
    result:list[UserInstrument] = []
    file_path: Path
    for file_path in path.glob("**/*.xml"):
        result.append(get_user_instrument(file_path))
    return result

def get_user_instrument(file_path:Path) -> UserInstrument:
    tree_root :ET.Element = ET.parse(file_path).getroot()
    params = dictionary_from_list_in_xml_node(tree_root)

    return UserInstrument(
        name=file_path.name,
        # population=params["instrumentPop"],
        id=get_text_of_a_child(tree_root, "id"),
        params=params,
    )

def get_text_of_a_child(element:ET.Element, tag:str) -> str:
    # This function mainly exists for the type checker
    x: ET.Element|None = element.find(tag)
    assert x is not None
    assert x.text is not None
    y:str = x.text
    return y 

def get_child(element:ET.Element, tag:str) -> ET.Element:
    # This function mainly exists for the type checker
    x: ET.Element|None = element.find(tag)
    assert x is not None
    return x 


def dictionary_from_list_in_xml_node(node: ET.Element) -> dict[str, str]:
    # input node is either list or has a child named list
    
    iterable = None
    if node.tag == "list":
        iterable = node
    else:
        iterable = node.find("list")

    if iterable == None:
        raise RuntimeError("Input node doesn't have a child named list, nor is itself a list")

    result: dict[str, str] = {} 
    for el in iterable:
        key = get_text_of_a_child(el, "keyName")
        value = None
        match get_text_of_a_child(el, "typeAString"):
            case "int":
                value = get_text_of_a_child(el, "valueAsInt")
            case "float":
                value = get_text_of_a_child(el, "valueAsFloat")
            case "long":
                value = get_text_of_a_child(el, "valueAsLong")
            case "double":
                value = get_text_of_a_child(el, "valueAsDouble")
            case "int":
                value = get_text_of_a_child(el, "valueAsInt")
            case "bool":
                value = get_text_of_a_child(el, "valueAsBoolean")
            case "String":
                value = get_text_of_a_child(el, "valueAsString")
            case _:
                raise RuntimeError("Undefined type in XML - " + get_text_of_a_child(el, "typeAString"))
        result[key] = value
    return result

def get_numerical_kernel_output(file_path:Path, instruments:list[UserInstrument]) -> NumericalResults:
    resulting_particle_detectors : list[ParticleDetector] = []
    for instrument in instruments:
        resulting_particle_detectors.append(get_particle_detector(path, instrument))



    return NumericalResults(
        emitted_currents= load_time_series(file_path / "emittedCurrents.txt"),
        number_of_superparticles=get_number_of_superparticles(file_path),
        particle_detectors=resulting_particle_detectors,
        time_steps=load_time_series(file_path / "Simulation_Control_-_time_steps_(s_._s)__TimeSteps.txt"),
        spis_log=(file_path / "SpisNum.log").read_text(encoding="utf_8", errors='backslashreplace'),
        total_current=load_time_series(file_path / "Total_current_on_spacecraft_surface._SCTotalCurrent.txt"),
        collected_currents=load_time_series(file_path / "collectedCurrents.txt"),
        surface_potential=load_time_series(file_path / "Average_surface_potential_of_node_0_(V_,_s)__ElecNode0_Potential.txt"),
    )

def load_time_series(path:Path) -> TimeSeries:
    print(path)
    return path

def load_mesh(path:Path) -> Mesh:
    pass

def get_particle_detector(path:Path, instrument:UserInstrument) -> ParticleDetector:
    pass

def get_number_of_superparticles(path:Path) -> list[NumberOfSuperparticles]:
    pass

if __name__=="__main__":
    log.basicConfig(level=2)
    # groups = get_groups(Path("C:\\Users\\matej\\Desktop\\VU\\8\\DefaultProject.spis5\\DefaultStudy\\Preprocessing\\Groups\\groups.xml"))
    # user_instruments = get_user_instruments(Path("C:\\Users\\matej\\Desktop\\VU\\8\\DefaultProject.spis5\\DefaultStudy\\Simulations\\Run1\\UserInstruments"))
    # path = Path("C:/Users/matej/Desktop/VU/8/DefaultProject.spis5")  / "DefaultStudy"
    path = Path("C:/Users/matej/Desktop/VU/example/example/cube_wsc_01.spis5")  / "CS_01"
    data = load_data(path)
    pass