import pathlib
import helpers
import reader
import logging as log
import default_settings
import electron_detector
import numpy as np
import configparser
import argparse

import matplotlib
matplotlib.use('TKAgg')
log.getLogger('matplotlib.font_manager').setLevel(log.ERROR)



@helpers.log_function_entry_and_exit
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--Input", help="Config file")
    args = parser.parse_args()

    if args.Input: 
        path_string = args.Input
    else:
        log.error("Missing config file path")
        path_string = "C:/temp/DP/SpisReader/example.config"


    config = configparser.ConfigParser()
    config.read(path_string)


    path = pathlib.Path(config["Simulation"]["path"])

    
    default_settings.Settings.print_current_settings()

    simulation = reader.load_simulation(path, force_processing=False)


    position = np.array([float(config["Detector"]["position_x"]),float(config["Detector"]["position_y"]),float(config["Detector"]["position_z"])])
    facing = np.array([float(config["Detector"]["facing_x"]),float(config["Detector"]["facing_y"]),float(config["Detector"]["facing_z"])])
    updirection = np.array([float(config["Detector"]["updirection_x"]),float(config["Detector"]["updirection_y"]),float(config["Detector"]["updirection_z"])])

    radius = float(config["Detector"]["radius"])

    acceptance_angle_phi = float(config["Detector"]["acceptance_angle_phi"])
    acceptance_angle_theha = float(config["Detector"]["acceptance_angle_theha"] )   
    
    number_of_samples_phi = int(config["Detector"]["number_of_samples_phi"])
    number_of_samples_theha = int(config["Detector"]["number_of_samples_theta"])

    max_number_of_steps = int(config["Detector"]["max_number_of_steps"])

    energy = float(config["Detector"]["energy"])


    boundary_temperature = float(config["Simulation"]["boundary_temperature"])

    detector = electron_detector.ElectronDetector(simulation,position=position, facing=facing, updirection=updirection, radius=radius, acceptance_angle_phi=acceptance_angle_phi, acceptance_angle_theta=acceptance_angle_theha,
                                                  number_of_samples_phi=number_of_samples_phi, number_of_samples_theta=number_of_samples_theha, max_number_of_steps=max_number_of_steps, energy=energy, boundary_temperature=boundary_temperature)



    print("Started energy " , energy) 
    detector.backtrack()
    assert default_settings.Settings.default_pickle_path is not None
    detector.save_self(default_settings.Settings.default_pickle_path / f"Detector_energy={energy}.pkl")
    print("Ended energy ", energy)





if __name__ == "__main__":
    helpers.default_log_config()
    main()
    log.info("Finished")
