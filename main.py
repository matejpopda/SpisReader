import reader
import pathlib
import logging as log

if __name__=="__main__":
    log.basicConfig(level=0)
    path = pathlib.Path("C:/Users/matej/Desktop/VU/8/DefaultProject.spis5")

    data = reader.load_data(path)
    
