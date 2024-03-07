import reader
import pathlib
import logging as log


if __name__=="__main__":
    log.basicConfig(level=0)
    path = pathlib.Path("C:/Users/matej/Desktop/VU/example/example/cube_wsc_01.spis5")  / "CS_01"

    data = reader.load_data(path)
    pass

    

# TODO: Make a JSON out of the data dictionary
# TODO: Make graphs
# TODO: One type of numerical output can be Channel1*, add handling, not sure why the files got created either

