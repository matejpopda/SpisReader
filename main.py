import reader
import pathlib

if __name__=="__main__":
    path = pathlib.Path("C:/Users/matej/Desktop/VU/8/DefaultProject.spis5")

    data = reader.load_data(path)
    
