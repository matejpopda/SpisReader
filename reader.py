import typing
from pathlib import Path
import xml.etree.ElementTree as ET

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
    def __init__(self) -> None:
        pass


class SimulationPreprocessing:
    """Data from preprocessing is stored here"""
    def __init__(self, path_to_preprocessing: Path) -> None:
        model = None # Loads file Preprocessing/Mesh/GeometricalSystem/model.msh 
        groups: list[Group] = get_groups(path_to_preprocessing / "Groups" / "groups.xml") 
        # \Preprocessing\Groups/groups.xml
        electrical_circuit : str = None # \Preprocessing\ElectricalCircuit\circuit.txt


class SimulationResults:
    """Results of a simulation (and variables that change per simulation, such as instruments are stored here)"""
    def __init__(self, path_to_results: Path) -> None:
        default_instruments : list[DefaultInstrument] = None #get_default_instruments(path_to_results / "DefaultInstruments")
        # List of instruments from \Simulations\Run1\DefaultInstruments

        user_instruments : list[UserInstrument] = None 
        # List of instruments from \Simulations\Run1\UserInstruments

        global_parameters = None 
        # \Simulations\Run1\GlobalParameters

        numerical_kernel_output = None 
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
        name = children.find("name").text
        SPISid = children.find("id").text
        GMSHid = children.find("linkedMeshMaskIds").find("int").text
        if children.find("type") is not None:
            type = children.find("type").text
            print(type)
            match type: 
                case "Computational volume group":
                    print("gdfgdfggd")



if __name__=="__main__":
    get_groups(Path("C:\\Users\\matej\\Desktop\\VU\\8\\DefaultProject.spis5\\DefaultStudy\\Preprocessing\\Groups\\groups.xml"))