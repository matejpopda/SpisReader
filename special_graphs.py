import electron_detector
import numpy as np
import scipy
import scipy.constants as const
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


def uniform_field_test(sim):
    dummy_detector = electron_detector.ElectronDetector(sim, np.array((-1, 0, 0)), 1, 1)
    dummy_detector.position = np.array((-1, 0, 0))
    dummy_detector.orientation = np.array((1, 0, 0))
    dummy_detector.radius = 1

    electrons_euler: list[electron_detector.Electron] = []
    electrons_rk: list[electron_detector.Electron] = []
    electrons_boris: list[electron_detector.Electron] = []

    def uniform_field(position):
        return np.array((0, 1, 0))

    deltatime = -0.7 / np.sqrt(2 * 5 * scipy.constants.eV / scipy.constants.electron_mass)

    for energy in range(1, 10, 2):
        dummy_detector.energy = energy

        electron_euler = dummy_detector.generate_electron_vector(np.array([1, 0, 0]), np.array((0, 0)))
        electron_boris = dummy_detector.generate_electron_vector(np.array([1, 0, 0]), np.array((0, 0)))
        electron_rk = dummy_detector.generate_electron_vector(np.array([1, 0, 0]), np.array((0, 0)))

        electrons_euler.append(electron_euler)
        electrons_rk.append(electron_rk)
        electrons_boris.append(electron_boris)

        # print(electron_euler.velocity)

        for i in range(10):
            electron_euler.append_position_to_history()
            electron_boris.append_position_to_history()
            electron_rk.append_position_to_history()
            electron_detector.euler_scheme(
                electron_euler, dt=deltatime, E=uniform_field(electron_euler.position)
            )
            electron_detector.boris_scheme(
                electron_boris, dt=deltatime, E=uniform_field(electron_boris.position)
            )
            electron_detector.rk_scheme(
                electron_rk,
                dt=deltatime,
                E=uniform_field(electron_rk.position),
                func_for_efield=uniform_field,
            )

    def plot_analytical_position(energy: float):
        x = []
        y = []
        smoothness = 10
        speed = np.sqrt(2 * energy * const.eV / const.electron_mass)
        for i in range(10 * smoothness):
            x.append(-speed * i * deltatime / smoothness)
            y.append(
                0.5 * (-const.elementary_charge / const.electron_mass) * (i * deltatime / smoothness) ** 2
            )
        plt.plot(x, y, c="black", label="Analytical solution")

    for i in range(1, 10, 2):
        plot_analytical_position(i)

    for e in electrons_euler:
        positions_e_x = [x[0] for x in e.position_history]
        positions_e_y = [x[1] for x in e.position_history]
        plt.plot(positions_e_x, positions_e_y, c="red")

    for e in electrons_rk:
        positions_e_x = [x[0] for x in e.position_history]
        positions_e_y = [x[1] for x in e.position_history]
        plt.plot(positions_e_x, positions_e_y, c="green")

    for e in electrons_boris:
        positions_e_x = [x[0] for x in e.position_history]
        positions_e_y = [x[1] for x in e.position_history]
        plt.plot(positions_e_x, positions_e_y, c="blue")

    plt.xlabel("X position [m]")
    plt.ylabel("Y position [m]")
    plt.title("Trajectory of particles in an uniform electric field")

    # Add manual legend
    legend_elements = [
        Line2D([0], [0], color="red", label="Euler scheme"),
        Line2D([0], [0], color="green", label="RK2 scheme"),
        Line2D([0], [0], color="blue", label="Boris scheme"),
        Line2D([0], [0], color="black", label="Analytical solution"),
    ]
    plt.legend(handles=legend_elements)

    plt.show()
