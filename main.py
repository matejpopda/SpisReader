import spisModule as spis
import pathlib
import helpers


@helpers.log_function_entry_and_exit
def main():
    
    path = pathlib.Path("C:/Users/matej/Desktop/VU/example/example/cube_wsc_01.spis5")  / "CS_01"

    spis.save_as_pickle(spis.load_from_SPIS(path), path /'processed_simulation.pkl')

    result = spis.load_pickle(path /'processed_simulation.pkl')

    print(result.results.extracted_data_fields.spacecraft_face.properties)

    print(result.results.extracted_data_fields.spacecraft_mesh.properties)
    print(result.results.extracted_data_fields.spacecraft_vertex.properties)
    print(result.results.extracted_data_fields.volume_vertex.properties)






if __name__=="__main__":
    helpers.default_log_config()
    main()
    

# TODO: Make a JSON out of the data dictionary
# TODO: Make graphs
# TODO: One type of numerical output can be Channel1*, add handling, not sure why the files got created either
# TODO: Ignoring Datafield monitored and extracted timeseries because its obtainable from masks