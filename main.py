import spisModule as spis
import pathlib
import spisModule.helpers as helpers
import spisModule.plotters as plotters


@helpers.log_function_entry_and_exit
def main():
    
    # path = pathlib.Path("C:/Users/matej/Desktop/VU/example/example/cube_wsc_01.spis5")  / "CS_01"
    path = pathlib.Path("C:/Users/matej/Desktop/VU/datafromsofie/S03_11.spis5/S03_11")

    result = spis.load_simulation(path)

    # print(result.results.extracted_data_fields.spacecraft_face.properties)

    # print(result.results.extracted_data_fields.spacecraft_mesh.properties)
    # print(result.results.extracted_data_fields.spacecraft_vertex.properties)
    # print(result.extracted_data_fields.volume_vertex.properties)

    # print(result.results.extracted_data_fields.spacecraft_face.properties)

    plotters.interactive_plot_physical_mesh(result.results.extracted_data_fields.spacecraft_mesh) 
    plotters.interactive_plot_physical_mesh(result.results.extracted_data_fields.spacecraft_face.mesh)
    # plotters.interactive_plot_physical_mesh(result.results.extracted_data_fields.spacecraft_vertex.mesh)
    # plotters.interactive_plot_physical_mesh(result.results.extracted_data_fields.volume_vertex.mesh)


    # plotters.save_mesh(result.extracted_data_fields.spacecraft_face, "Conductance_t=0.0s")
    # plotters.slice_and_save(result.extracted_data_fields.volume_vertex, "final_elec1_charge_density_-_step0", normal=plotters.PlaneNormals.XZ)

    # plotters.interactive_plot_orth_slice(result.results.extracted_data_fields.volume_vertex, "final_elec1_charge_density_-_step0")

    


if __name__=="__main__":
    helpers.default_log_config()
    main()
    

# TODO: Make a JSON out of the data dictionary
# TODO: One type of numerical output can be Channel1*, add handling, not sure why the files got created either
# TODO: Ignoring Datafield monitored and extracted timeseries because its obtainable from masks