import typing
from pathlib import Path
import xml.etree.ElementTree as ET
import logging as log

class DefaultInstrument:
    """Class encapsulating a single instrument"""
    def __init__(self) -> None:
        pass


class UserInstrument:
    """Class encapsulating a single instrument"""
    def __init__(self) -> None:
        pass


class Group:
    """Class encapsulating a single physical group"""
    def __init__(self,*, name:str, SPISid:str, GMSHid:str, type:str, properties:list["GroupProperty"]) -> None:
        self.name:str = name
        self.SPISid:str = SPISid
        self.GMSHid:str = GMSHid
        self.type:str = type
        self.properties:list[GroupProperty] = properties

class GroupProperty:
    """Class encapsulating group properties"""
    def __init__(self, *, name:str, id:str, description:str) -> None:
        self.name:str = name
        self.id:str = id
        self.description:str = description

class SimulationPreprocessing:
    """Data from preprocessing is stored here"""
    def __init__(self, path_to_preprocessing: Path) -> None:
        self.model = None 
        # Loads file Preprocessing/Mesh/GeometricalSystem/model.msh 

        self.groups: list[Group] = get_groups(path_to_preprocessing / "Groups" / "groups.xml") 
        # \Preprocessing\Groups/groups.xml

        self.electrical_circuit : str = (path_to_preprocessing / "ElectricalCircuit" / "circuit.txt").read_text() 
        # \Preprocessing\ElectricalCircuit\circuit.txt


class SimulationResults:
    """Results of a simulation (and variables that change per simulation, such as instruments are stored here)"""
    def __init__(self, path_to_results: Path) -> None:
        self.default_instruments : list[DefaultInstrument] = None #get_default_instruments(path_to_results / "DefaultInstruments")
        # List of instruments from \Simulations\Run1\DefaultInstruments

        self.user_instruments : list[UserInstrument] = None 
        # List of instruments from \Simulations\Run1\UserInstruments

        self.global_parameters = None 
        # \Simulations\Run1\GlobalParameters

        self.numerical_kernel_output = None 
        # \Simulations\Run1\NumKernel\Output


class Simulation:
    """Wrapper for preprocessing information and results"""
    def __init__(self, preprocessing, results) -> None:
        self.preprocessing : SimulationPreprocessing = preprocessing
        self.results : SimulationResults = results
      

def load_data(path: Path) -> Simulation:
    
    results = SimulationResults(path / "DefaultStudy" / "Simulations" / "Run1" )
    preprocessing = SimulationPreprocessing(path / "DefaultStudy" / "Preprocessing")

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


if __name__=="__main__":
    log.basicConfig(level=0)
    groups = get_groups(Path("C:\\Users\\matej\\Desktop\\VU\\8\\DefaultProject.spis5\\DefaultStudy\\Preprocessing\\Groups\\groups.xml"))
    pass