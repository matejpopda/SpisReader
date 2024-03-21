import spisModule as spis
import pathlib
import logging as log

if __name__=="__main__":

    logger = log.getLogger()
    logger.setLevel(log.DEBUG)


    path = pathlib.Path("C:/Users/matej/Desktop/VU/example/example/cube_wsc_01.spis5")  / "CS_01"
    
    spis.save_as_pickle(spis.load_from_SPIS(path), path /'processed_simulation.pkl')

    result = spis.load_pickle(path /'processed_simulation.pkl')



    pass

    

# TODO: Make a JSON out of the data dictionary
# TODO: Make graphs
# TODO: One type of numerical output can be Channel1*, add handling, not sure why the files got created either

