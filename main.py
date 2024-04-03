import spisModule as spis
import pathlib
import helpers
import pyvista
import pyvista.plotting

@helpers.log_function_entry_and_exit
def main():
    
    # path = pathlib.Path("C:/Users/matej/Desktop/VU/example/example/cube_wsc_01.spis5")  / "CS_01"
    path = pathlib.Path("C:/Users/matej/Desktop/VU/datafromsofie/S03_11.spis5/S03_11")

    result = spis.load_simulation(path, force_raw_processing=True)

    # print(result.results.extracted_data_fields.spacecraft_face.properties)

    # print(result.results.extracted_data_fields.spacecraft_mesh.properties)
    # print(result.results.extracted_data_fields.spacecraft_vertex.properties)
    # print(result.results.extracted_data_fields.volume_vertex.properties)

    print(result.results.extracted_data_fields.spacecraft_face.properties)

    plotter = pyvista.plotting.Plotter()
    plotter.add_mesh(result.results.extracted_data_fields.spacecraft_face.mesh, 
                     scalars="gmsh:physical",
                     )

    plotter.show()
    plotter = pyvista.plotting.Plotter()
    plotter.add_mesh(result.results.extracted_data_fields.spacecraft_face.mesh, 
                     scalars="Conductance_t=0.0s",
                     )

    plotter.show()


if __name__=="__main__":
    helpers.default_log_config()
    main()
    

# TODO: Make a JSON out of the data dictionary
# TODO: Make graphs
# TODO: One type of numerical output can be Channel1*, add handling, not sure why the files got created either
# TODO: Ignoring Datafield monitored and extracted timeseries because its obtainable from masks