import pathlib
import helpers 
import plotters 
import reader
import logging as log

@helpers.log_function_entry_and_exit
def main():
    
    # path = pathlib.Path("C:/Users/matej/Desktop/VU/example/example/cube_wsc_01.spis5")  / "CS_01"
    path = pathlib.Path("C:/Users/matej/Desktop/VU/datafromsofie/S03_11.spis5/S03_11")

    result = reader.load_simulation(path)

    # print(result.results.extracted_data_fields.spacecraft_face.properties)

    # print(result.results.extracted_data_fields.spacecraft_mesh.properties)
    # print(result.results.extracted_data_fields.spacecraft_vertex.properties)
    # print(result.extracted_data_fields.volume_vertex.properties)



    # print(result.results.extracted_data_fields.spacecraft_face.properties)
    # plotters.interactive_plot_physical_mesh(result.results.extracted_data_fields.spacecraft_mesh) 
    # plotters.interactive_plot_physical_mesh(result.results.extracted_data_fields.spacecraft_face.mesh)
    # plotters.interactive_plot_physical_mesh(result.results.extracted_data_fields.spacecraft_vertex.mesh)
    # plotters.interactive_plot_physical_mesh(result.results.extracted_data_fields.volume_vertex.mesh)

    # plotters.save_mesh(result.extracted_data_fields.spacecraft_face, "Conductance_t=0.0s")
    # plotters.slice_and_save(result.extracted_data_fields.volume_vertex, "final_elec1_charge_density_-_step0", normal=plotters.PlaneNormals.XZ)
    # plotters.xz_slice(result.extracted_data_fields.volume_vertex, "final_elec1_charge_density_-_step0")
    # plotters.interactive_plot_orth_slice(result.results.extracted_data_fields.volume_vertex, "final_elec1_charge_density_-_step0")

    total_charge =  plotters.glob_properties(result, "improved__total_charge_density_at_t_=_*")
    log.info("started plotting gif of size " + str(len(total_charge)))
    plotters.make_gif_xz_slice(total_charge, "total_charge")
    log.info("stopped plotting gif")

    log.info("started plotting images")
    for i, j in plotters.glob_properties(result.results.extracted_data_fields.volume_vertex, "improved__total_charge_density_at_t_=_*"):
        plotters.xz_slice(i, j)
    log.info("stopped plotting images")


if __name__=="__main__":
    helpers.default_log_config()
    main()
    

# TODO: Make a JSON out of the data dictionary
# TODO: One type of numerical output can be Channel1*, add handling, not sure why the files got created either
# TODO: Ignoring Datafield monitored and extracted timeseries because its obtainable from masks