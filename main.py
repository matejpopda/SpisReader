import pathlib
import helpers
import plotters
import reader
import logging as log
import utils
import default_settings
import electron_detector

import multiprocessing

import pickle

import pyvista
import pyvista.plotting.plotter

import matplotlib
matplotlib.use('TKAgg')
log.getLogger('matplotlib.font_manager').setLevel(log.ERROR)


def run_backtrack(detector: electron_detector.ElectronDetector, energy):
    print("Started energy " , energy)
    detector.backtrack()
    detector.simulation = None
    detector.save_self(default_settings.Settings.default_pickle_path / ("Detector_" + str(energy) + ".pkl"))
    print("Ended energy ", energy)

@helpers.log_function_entry_and_exit
def main():
    path = pathlib.Path("C:/Users/matej/Desktop/VU/data_efield/sofiedata/SOLO06.spis5/SOLO06")

    default_settings.Settings.print_current_settings()

    result = reader.load_simulation(path, force_processing=False)

    mesh = plotters.glob_properties(result, "*spacecraft*")[0][0]

    utils.generate_efield_vector_property(result)


    detectors: list[electron_detector.ElectronDetector] = []
    processes: list[multiprocessing.Process] = []

    # energies = [1,2,3,4,5,6,7,8,12,16,24,32,64,120]
    # energies = [1,2]
    # energies = [64]
    # energies = [1,3,5,6,7,8,12,16,24,32,64,120]
    energies = [1,3,5,12,32,64,120]
    # energies = [1]

    if True:
        for energy in energies:
            detector = electron_detector.ElectronDetector(result, energy=energy)
            # detector.number_of_steps = 3
            detector.number_of_samples_phi = 20
            detector.number_of_samples_theta = 20
            detectors.append(detector)

            process = multiprocessing.Process(target=run_backtrack, args=(detector,energy,))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()


    detectors.clear()
    for energy in energies:
        save_path = default_settings.Settings.default_pickle_path / f"Detector_{energy}.pkl"
        with open(save_path, 'rb') as f:
            detector = pickle.load(f)
        detectors.append(detector)


    # plotters.interactive_plot_mesh_with_typed_trajectories(mesh, [])

    for detector in detectors:
        print(detector.energy)
        # plotters.interactive_plot_mesh_with_typed_trajectories(mesh, detector.get_typed_trajectories())
        # plotters.interactive_plot_electron_detectors(mesh, [detector])

    # trajectories = [x.data for x in result.extracted_data_fields.particle_trajectories]
    # plotters.interactive_plot_mesh_with_trajectories(mesh, trajectories)


        # detector.result_accumulator.plot()

    # plotters.plot_final_quantities(result)

    plotters.interactive_plot_electron_detectors(mesh, detectors)

    plotters.detectors_to_1d_distribution(detectors)

    exit()

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


    #### Drawing normals
    # plotter = pyvista.plotting.plotter.Plotter()
    # surf =   mesh.mesh.extract_surface()
    # plotter.add_mesh(surf)
    # arrows = surf.point_normals
    # plotter.add_arrows(surf.points, arrows)
    # plotter.show()




if __name__ == "__main__":
    helpers.default_log_config()
    main()
    log.info("Finished")
