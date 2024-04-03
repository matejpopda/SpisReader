import spisModule as spis
import pathlib
import logging as log
import sys


if __name__=="__main__":
    log.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s -  %(message)s', 
                    level=log.DEBUG, 
                    filename="latest_run.log",
                    filemode='w')
    handler = log.StreamHandler(sys.stdout)
    log.getLogger().addHandler(handler)
    handler.setLevel(log.INFO)
    handler.setFormatter(log.Formatter(fmt = '%(asctime)s - %(levelname)s - %(filename)s -  %(message)s'))

    log.info("Program started")

    path = pathlib.Path("C:/Users/matej/Desktop/VU/example/example/cube_wsc_01.spis5")  / "CS_01"


    spis.reader.get_extracted_datafields(path / "Simulations/Run1/OutputFolder/DataFieldExtracted")
    exit()



    spis.save_as_pickle(spis.load_from_SPIS(path), path /'processed_simulation.pkl')

    result = spis.load_pickle(path /'processed_simulation.pkl')


    

# TODO: Make a JSON out of the data dictionary
# TODO: Make graphs
# TODO: One type of numerical output can be Channel1*, add handling, not sure why the files got created either

