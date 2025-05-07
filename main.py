import pathlib
import helpers
import plotters
import reader
import logging as log
import utils
import default_settings
import electron_detector
import simulation
import numpy as np
import matplotlib.pyplot as plt 
import multiprocessing
import spisutils
import pickle
import scipy.constants as const

import pyvista
import pyvista.plotting.plotter

import matplotlib
matplotlib.use('TKAgg')
log.getLogger('matplotlib.font_manager').setLevel(log.ERROR)


def run_backtrack(detector: electron_detector.ElectronDetector, energy: float, bt_type: electron_detector.BacktrackingTypes):
    print("Started energy " , energy, " backtracking type ", bt_type.name) 
    detector.backtracking_type = bt_type
    detector.backtrack()
    detector.simulation = None
    detector.save_self(default_settings.Settings.default_pickle_path / f"Detector_energy={energy}_bttype={bt_type.name}.pkl")
    print("Ended energy ", energy, " backtracking type ", bt_type.name)


def simulate_1d_detector(sim: simulation.Simulation, force_sim: bool = False, bt_type: electron_detector.BacktrackingTypes = electron_detector.BacktrackingTypes.Euler):

    save_path = default_settings.Settings.default_pickle_path / f"1D_detector_{bt_type.name}.pkl"

    if save_path.exists() and not force_sim: 
        with open(save_path, 'rb') as f:
            detector = pickle.load(f)
        return detector


    detector_1d = electron_detector.ElectronDetector(sim, energy=1)
    detector_1d.number_of_samples_phi = 1
    detector_1d.number_of_samples_theta = 1
    detector_1d.number_of_steps = 500
    detector_1d.backtracking_type = bt_type
    detector_1d.orientation = np.array([1, 0, 0])
    detector_1d.acceptance_angle_phi = 0.000001
    detector_1d.acceptance_angle_theta = 0.0000001

    log.info(f"Simulating 1D_detector_{bt_type.name}")

    for i in range(2,120):
        detector_1d.energy = i
        detector_1d.calculate_dt()
        detector_1d.backtrack()
    detector_1d.simulation = None
    detector_1d.save_self(save_path)
    return detector_1d


def uniform_field_test(sim):
    dummy_detector = electron_detector.ElectronDetector(sim, 10)
    dummy_detector.position = np.array((-1,0,0))
    dummy_detector.orientation = np.array((1,0,0))
    dummy_detector.radius = 1

    electrons_euler: list[electron_detector.Electron] = []
    electrons_rk   : list[electron_detector.Electron] = []
    electrons_boris: list[electron_detector.Electron] = []

    def uniform_field(position):
        return np.array((0,1,0))
    
    
    deltatime = - 0000.1


    for energy in range(1,10,2):
        dummy_detector.energy = energy

        electron_euler = dummy_detector.generate_electron_vector([1,0,0], (0,0))
        electron_boris = dummy_detector.generate_electron_vector([1,0,0], (0,0))
        electron_rk = dummy_detector.generate_electron_vector([1,0,0], (0,0))

        electrons_euler.append(electron_euler)
        electrons_rk.append(electron_rk)
        electrons_boris.append(electron_boris)

        # print(electron_euler.velocity)   

        for i in range(10):
            electron_euler.append_position_to_history()
            electron_boris.append_position_to_history()
            electron_rk.append_position_to_history()
            electron_detector.euler_scheme(electron_euler, dt=deltatime, E = uniform_field(electron_euler.position))
            electron_detector.boris_scheme(electron_boris, dt=deltatime, E = uniform_field(electron_boris.position))
            electron_detector.rk_scheme(electron_rk, dt=deltatime, E = uniform_field(electron_rk.position), func_for_efield=uniform_field)


    for e in electrons_euler:
        positions_e_x = [x[0] for x in  e.position_history]
        positions_e_y = [x[1] for x in  e.position_history]
        plt.plot(positions_e_x, positions_e_y, c="red", label="Euler scheme")
        
    for e in electrons_rk:
        positions_e_x = [x[0] for x in  e.position_history]
        positions_e_y = [x[1] for x in  e.position_history]
        plt.plot(positions_e_x, positions_e_y, c="green", label="RK2 scheme")
        
    for e in electrons_boris:
        positions_e_x = [x[0] for x in  e.position_history]
        positions_e_y = [x[1] for x in  e.position_history]
        plt.plot(positions_e_x, positions_e_y, c="blue", label="Boris scheme")

    def plot_analytical_position(energy:float):
        x = []
        y = []
        smoothness = 10
        speed = np.sqrt(2 * energy * const.eV/ const.electron_mass)
        for i in range(10 * smoothness):
            x.append(speed * i * deltatime/smoothness)
            y.append(0.5 * (-const.elementary_charge/ const.electron_mass) * (i * deltatime/smoothness) ** 2 )
        plt.plot(x, y, c="black", label="Analytical solution")


    for i in range(1, 10,2):
        plot_analytical_position(i)

    plt.show()






@helpers.log_function_entry_and_exit
def main():




    path = pathlib.Path("C:/temp/DP/SOLO06.spis5/SOLO06")
    # path = pathlib.Path("C:/temp/DP-sim/SOLOA14/SOLOA14.spis5/SOLOA14")


    
    default_settings.Settings.print_current_settings()

    result = reader.load_simulation(path, force_processing=False)

    # uniform_field_test(result)
    # exit()

    for detector in result.results.numerical_kernel_output.particle_detectors:
        for plist in detector.particle_list:
            reader.load_unloaded_particle_list(plist)

    mesh = plotters.glob_properties(result, "*spacecraft*")[0][0]

    utils.generate_efield_vector_property(result)

    # particle_list = spisutils.get_particle_list(result)
    # spisutils.plot_pl_EDF(particle_list)

    detector_1d_euler = simulate_1d_detector(result, force_sim=False, bt_type = electron_detector.BacktrackingTypes.Euler)
    detector_1d_boris = simulate_1d_detector(result, force_sim=False, bt_type = electron_detector.BacktrackingTypes.Boris)
    detector_1d_rk = simulate_1d_detector(result, force_sim=False, bt_type = electron_detector.BacktrackingTypes.RK)
    plotters.plot_detectors_with_0_acceptance_angle([detector_1d_boris, detector_1d_euler, detector_1d_rk]) 
    # plotters.plot_detectors_with_0_acceptance_angle([detector_1d_boris]) 
    plotters.plot_detectors_with_0_acceptance_angle([detector_1d_euler]) 

    plotters.interactive_plot_electron_detectors_differentiate_detectors_by_color(mesh, [detector_1d_euler])
    plotters.interactive_plot_electron_detectors_differentiate_detectors_by_color(mesh, [detector_1d_boris, detector_1d_euler, detector_1d_rk])

    detectors: list[electron_detector.ElectronDetector] = []
    processes: list[multiprocessing.Process] = []

    # energies = [1,2,3,4,5,6,7,8,12,16,24,32,64,120]
    # energies = [1,2]
    # energies = [64]
    energies = [1,3,5,6,7,8,12,16,24,32,64,120]
    # energies = [1,3,5,12,32,64,120]
    # energies = [1]
    # energies = [2,4,8, 50]
    # energies = [50]
    # energies = [0.5,1,1.5,2,2.5,3,4,5,6,7,8,12,16,24,32,64,85,120]
    # energies = [0.5,1,1.5,2,2.5,3,4,5,6,7,8,12,64,85,120]
    # energies = [2,2.5,3,4,5,6,7,8,12,64,85,120]
    # energies = [4,8,12,64]
    energies = [64]
    energies = [63]
    # energies = [50]
    # energies = [120]

    backtrack_types = [electron_detector.BacktrackingTypes.Boris, electron_detector.BacktrackingTypes.Euler, electron_detector.BacktrackingTypes.RK]
    # backtrack_types = [electron_detector.BacktrackingTypes.Boris, electron_detector.BacktrackingTypes.Euler]
    # backtrack_types = [electron_detector.BacktrackingTypes.Boris]
    # backtrack_types = [electron_detector.BacktrackingTypes.Euler]
    backtrack_types = [electron_detector.BacktrackingTypes.RK]

    # if True:
    #     for bt_type in backtrack_types:
    #         for energy in energies: 
    #             detector = electron_detector.ElectronDetector(result, energy=energy)
    #             detector.number_of_steps = 30
    #             detector.number_of_samples_phi = 20
    #             detector.number_of_samples_theta = 20
    #             detectors.append(detector)

    #             process = multiprocessing.Process(target=run_backtrack, args=(detector,energy, bt_type,))
    #             process.start()
    #             processes.append(process)

    #     for process in processes:
    #         process.join()


    detectors.clear()
    for energy in energies:
        for bt_type in backtrack_types: 
            save_path = default_settings.Settings.default_pickle_path / f"Detector_energy={energy}_bttype={bt_type.name}.pkl"
            with open(save_path, 'rb') as f:
                detector = pickle.load(f)
            detectors.append(detector)




    # plotters.interactive_plot_mesh_with_typed_trajectories(mesh, [])
    # plotters.interactive_plot_electron_detectors_differentiate_detectors_by_color(mesh, detectors)

    # trajectories = [x.data for x in result.extracted_data_fields.particle_trajectories]
    # plotters.interactive_plot_mesh_with_trajectories(mesh, trajectories)

    for detector in detectors:
        print(detector.energy)
        # plotters.interactive_plot_mesh_with_typed_trajectories(mesh, detector.get_typed_trajectories())
        plotters.interactive_plot_electron_detectors(mesh, [detector])




        detector.result_accumulator.plot()

    # plotters.plot_final_quantities(result)

    # plotters.interactive_plot_electron_detectors(mesh, detectors)

    # plotters.detectors_to_1d_distribution_bad(detectors)


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
