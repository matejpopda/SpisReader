import pathlib
import helpers
import plotters
import reader
import logging as log

import default_settings



import matplotlib
matplotlib.use('TKAgg')

@helpers.log_function_entry_and_exit
def main():
    path = pathlib.Path("C:/Users/matej/Desktop/VU/example/example/cube_wsc_01.spis5") / "CS_01"



    default_settings.Settings.print_current_settings()


    result = reader.load_simulation(path, force_processing=False)

    plotters.plot_final_quantities(result)

    total_charge = plotters.glob_properties(result, "improved__total_charge_density_at_t_=_*")
    log.info("started plotting gif of size " + str(len(total_charge)))
    plotters.make_gif_xz_slice(total_charge, "total_charge")
    log.info("stopped plotting gif")

    total_charge = plotters.glob_properties(result, "improved_ions1_charge_density_at_t_=*")
    log.info("started plotting gif of size " + str(len(total_charge)))
    plotters.make_gif_xz_slice(total_charge, "ions1_charge_density")
    log.info("stopped plotting gif")

    total_charge = plotters.glob_properties(result, "improved_net_current__on_regular_surf_at_t_=*")
    log.info("started plotting gif of size " + str(len(total_charge)))
    plotters.make_gif_surface_from_default_view(total_charge, "net_current")
    log.info("stopped plotting gif")

    total_charge = plotters.glob_properties(result, "plasma_pot_at_t_=_*")
    log.info("started plotting gif of size " + str(len(total_charge)))
    plotters.make_gif_xz_slice(total_charge, "plasma_pot")
    log.info("stopped plotting gif")



if __name__ == "__main__":
    helpers.default_log_config()
    main()
