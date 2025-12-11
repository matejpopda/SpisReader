import simulation
from typing import Tuple, Any, TypeAlias, ClassVar, Literal
from dataclasses import dataclass
import dataclasses
import scipy.constants # type: ignore
import numpy as np
import random
from enum import Enum
import utils
import matplotlib.pyplot as plt
import scipy.constants as consts
import pandas as pd 
import logging as log
import pickle
from multiprocessing import Pool

import default_settings
import numpy.typing as npt
import typing

Vector3D: TypeAlias = npt.NDArray[np.floating[typing.Any]]
Vector2D: TypeAlias = npt.NDArray[np.floating[typing.Any]]


DEFAULT_PROJECTION_DIRECTION: Vector3D = np.array([1,0,0])


MAX_PROCESSES = 5

DETECTOR: "ElectronDetector" = None

def init_worker(detector_instance):
    print("Initializing worker")
    global DETECTOR
    DETECTOR = detector_instance

def backtrack_worker(idx):
    print(f"Started backtracking {idx} electron")
    electron = DETECTOR.particles[idx]
    DETECTOR.backtrack_one_electron(electron)
    print(f"Done backtracking {idx} electron")
    return electron  


class CollisionTypes(Enum):
    No_collision = 0
    Spacecraft = 1 
    Boundary = 2 
    Too_many_steps = 3 

class BacktrackingTypes(Enum):
    Euler = 0
    Boris = 1 
    RK = 2 

@dataclass
class Electron:
    MASS: ClassVar[float] = scipy.constants.electron_mass
    CHARGE: ClassVar[float] = scipy.constants.elementary_charge
    

    position: Vector3D
    velocity: Vector3D

    previous_position: Vector3D
    previous_velocity: Vector3D

    origin: Vector2D # Local coordinate to the sensor
    starting_energy: float

    collision_type: CollisionTypes = CollisionTypes.No_collision

    position_history: list[Vector3D] = dataclasses.field(default_factory=list)
    potential_history: list[float] = dataclasses.field(default_factory=list)
    energy_history : list[float] = dataclasses.field(default_factory=list)
    dt_history : list[float] = dataclasses.field(default_factory=list)
    

    number_steps_taken: int = 0

    probability_ambient: float|None = None
    probability_photo: float|None = None
    probability_secondary: float|None = None

    count_ambient: float | None = None
    count_photo: float | None = None
    count_secondary: float| None = None

    def append_position_to_history(self):
        self.position_history.append(self.position)

    def get_energy(self):
        speed_squared = self.velocity[0]**2 + self.velocity[1]**2 + self.velocity[2]**2 

        energy = 0.5 * speed_squared * self.MASS

        

        return energy / scipy.constants.value("electron volt-joule relationship")




class ResultAccumulator:
    def __init__(self) -> None:

        self.particles: list[Electron] = []

    def add_particle(self, electron: Electron):
        self.particles.append(electron)
    
    def plot_energies_against_probability(self):
        for particle in self.particles:
            if particle.probability_ambient is not None: 
                plt.scatter(particle.starting_energy, particle.probability_ambient)
        plt.show()


    def plot(self):
        
        # find global min max
        minimum = 10
        maximum = 0
        for i in self.particles:
            for current_val in [i.probability_ambient, i.probability_photo, i.probability_secondary]:
                if current_val is None:
                    break
                minimum = min(minimum, current_val)
                maximum = max(maximum, current_val)

        ambient_xx  = []
        ambient_yy = []
        ambient_cc = []
        for i in self.particles:
            ambient_xx.append(i.origin[0])
            ambient_yy.append(i.origin[1])
            if i.probability_ambient is not None:
                ambient_cc.append(i.probability_ambient)
            else: 
                ambient_cc.append(0)        

        plt.tripcolor(ambient_xx,ambient_yy,ambient_cc, cmap="viridis")
        plt.title("3D distribution function for ambient electrons")
        plt.ylabel("Elevation angle [rad]")
        plt.xlabel("Azimuthal angle [rad]")
        plt.colorbar().set_label("VDF [$m^{-6}s^3$]")
        plt.show()


        see_xx  = []
        see_yy = []
        see_cc = []
        for i in self.particles:
            see_xx.append(i.origin[0])
            see_yy.append(i.origin[1])
            if i.probability_secondary is not None:
                see_cc.append(i.probability_secondary)
            else: 
                see_cc.append(0)        

        plt.tripcolor(see_xx,see_yy,see_cc, cmap="viridis")
        plt.title("3D distribution function for secondary electrons")
        plt.ylabel("Elevation angle [rad]")
        plt.xlabel("Azimuthal angle [rad]")
        plt.colorbar().set_label("VDF [$m^{-6}s^3$]")
        plt.show()


        photo_xx  = []
        photo_yy = []
        photo_cc = []
        for i in self.particles:
            photo_xx.append(i.origin[0])
            photo_yy.append(i.origin[1])
            if i.probability_photo is not None:
                photo_cc.append(i.probability_photo)
            else: 
                photo_cc.append(0) 

        plt.tripcolor(photo_xx,photo_yy,photo_cc, cmap="viridis")
        plt.title("3D distribution function for photoelectrons")
        plt.ylabel("Elevation angle [rad]")
        plt.xlabel("Azimuthal angle [rad]")
        plt.colorbar().set_label("VDF [$m^{-6}s^3$]")
        plt.show()




class ElectronDetector:
    def __init__(self, data: simulation.Simulation, position:Vector3D, radius:float,  energy:float, facing:Vector3D = np.array([-1, 0, 0]),updirection:Vector3D = np.array([0, 1 ,0])
                 , acceptance_angle_phi: float = np.pi * 2 ,acceptance_angle_theta: float = np.pi, number_of_samples_theta: int = 5, number_of_samples_phi: int = 5, max_number_of_steps: int = 10, boundary_temperature: float = 103000 ) -> None:
        self.particles: list[Electron] = []
        self.position: Vector3D = position
        self.orientation: Vector3D = facing
        self.updirection: Vector3D = updirection
        self.radius: float = radius
        self.acceptance_angle_phi: float = acceptance_angle_phi
        self.acceptance_angle_theta: float = acceptance_angle_theta
        self.backtracking_type: BacktrackingTypes = BacktrackingTypes.RK


        ## 2, 14, 50 in eV
        self.energy : float = energy

        self.time: float

        
        # self.dt: float = 12 / (18755372 * (1))
        self.dt: float = self.calculate_dt()


        self.number_of_samples_theta: int = number_of_samples_theta
        self.number_of_samples_phi: int = number_of_samples_phi
        
        self.number_of_steps: int = max_number_of_steps


        self.simulation: simulation.Simulation = data

        self.result_accumulator : ResultAccumulator = ResultAccumulator()


        # probability calc
        self.boundary_temperature = boundary_temperature #for 8.9 eV its 103000



        # Settings for random drawing
        self.monte_carlo = False
        self.number_of_samples: int = 150
        

        # normalize orientation vectors
        self.orientation = utils.normalize(self.orientation)
        self.updirection = utils.normalize(self.updirection)

        self.e_field_mesh = utils.generate_efield_vector_property(self.simulation) # Contains "vector_electric_field"
        density_val = utils.glob_properties(self.simulation, "final_elec1_charge_density*")
        utils.glob_properties(self.simulation, "*final_photoElec_charge_density*")
        utils.glob_properties(self.simulation, "*final_secondElec_BS*")
        potential_val = utils.glob_properties(self.simulation, "*final_plasma_potential*")
        

        self.potential_name = potential_val[0][1]
        self.plasma_potential_mesh = potential_val[0][0]

        assert len(density_val) == 1 
        self.density_name = density_val[0][1]
        self.mesh = density_val[0][0]


        potential_val = utils.glob_properties(self.simulation, "final_plasma_potential*")
        assert len(potential_val) == 1 
        self.potential_name = potential_val[0][1]
        self.electric_field_from_potential = potential_val[0][0].mesh.compute_derivative(scalars=self.potential_name)


    def get_trajectories(self):
        result: list[list[Vector3D]] = []
        for particle in self.particles:
            result.append(particle.position_history)
        return result

    def calculate_dt(self):
        self.dt =  0.7 / np.sqrt(2 * self.energy * scipy.constants.eV / scipy.constants.electron_mass)
        return self.dt


    def get_typed_trajectories(self):
        result: list[Tuple[list[Vector3D], CollisionTypes]] = []
        for particle in self.particles:
            result.append(( particle.position_history, particle.collision_type))
        return result


    def generate_electrons_monte_carlo(self):
        for i in range(self.number_of_samples):
            if i % 50 == 0:
                print("sample", i, " from ", self.number_of_samples)

            electron = self.generate_electron()

        return 
    
    def save_self(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def generate_electrons_grid(self):
        

        def normalize(v: Vector3D) -> Vector3D:
            return v / np.linalg.norm(v)

        forward = normalize(self.orientation)
        right = normalize(np.cross(forward, self.updirection))
        true_up = np.cross(right, forward)


        thetas = np.linspace(-self.acceptance_angle_theta/2, self.acceptance_angle_theta/2, self.number_of_samples_theta)
        phis = np.linspace(-self.acceptance_angle_phi/2, self.acceptance_angle_phi/2, self.number_of_samples_phi)

        directions = []
        for phi in phis:
            row = []
            for theta in thetas:
                # Local spherical direction in camera space
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(phi)
                z = np.cos(theta) * np.cos(phi)
                
                # Rotate into world space using camera basis
                world_dir = (
                    x * right +
                    y * true_up +
                    z * forward
                )
                world_dir = normalize(world_dir)
                row.append(world_dir)
            directions.append(row)

        for n, x in enumerate(directions):
            # print(f"Backtracking: row {n+1},  out of  {self.number_of_samples_phi} for energy {self.energy}")
            for m, y in enumerate(x): 
                # print("Backtracking: column", m+1, " out of ", self.number_of_samples_theta)


                electron = self.generate_electron_vector(y, (thetas[m], phis[n])) #type: ignore 

                # self.backtrack_one_electron(electron)

                # if electron.collision_type != CollisionTypes.No_collision:
                #     self.accumulate_collision(electron)

        return 




    def backtrack(self):
        if self.monte_carlo == True:
            self.generate_electrons_monte_carlo()
        else:
            self.generate_electrons_grid()



        with Pool(
            processes=MAX_PROCESSES,
            initializer=init_worker,
            initargs=(self,)
        ) as pool:
            updated_particles = list(pool.imap_unordered(backtrack_worker, range(len(self.particles)), max(1, len(self.particles) // MAX_PROCESSES)))

        self.particles = updated_particles
        for electron in self.particles:
            if electron.collision_type != CollisionTypes.No_collision:
                    self.accumulate_collision(electron)


        

    def backtrack_one_electron(self, electron: Electron):
            collided = False
            step = 1
            self.move_backwards(electron, self.dt/1000) # Initializing the electron

            electron.position_history.append(electron.position)
            while not collided:

                step += 1 

                try:
                    self.move_backwards(electron, self.dt)
                except Exception:
                    collided = True
                    electron.collision_type = CollisionTypes.Boundary


                if self.check_boundary_collision(electron):
                    collided = True
                    electron.collision_type = CollisionTypes.Boundary

                SC_collisions = self.detect_collision_SC(electron)
                if len(SC_collisions) > 0:
                    collided = True
                    electron.position = self.get_position_after_collision(SC_collisions, electron)


                    electron.collision_type = CollisionTypes.Spacecraft
                if step > self.number_of_steps:
                    collided = True
                    electron.collision_type = CollisionTypes.Too_many_steps

                electron.position_history.append(electron.position)
            
            self.calculate_probability(electron)

    def get_position_after_collision(self, collisions: "list[int]", electron: Electron) -> Vector3D:
        spacecraft_mesh = self.simulation.results.extracted_data_fields.spacecraft_face.mesh
        
        distance = 1000000
        result = None

        assert len(collisions) > 0
        for id in collisions:
            # spacecraft_mesh.extract_cells(id)
            point = spacecraft_mesh.cell_centers().points[id]
            norm = np.linalg.norm(point - electron.previous_position)
            if norm < distance:
                distance = norm
                result = point


        # print(result)
        reverse_dir = np.array(electron.position) - np.array(electron.previous_position)
        reverse_offset = reverse_dir/np.linalg.norm(reverse_dir) * 0.0001

        result =  result + reverse_offset

        # print(result)
        return result + reverse_offset
            



    def calculate_probability(self, electron: Electron) -> None:
        if electron.collision_type == CollisionTypes.Boundary:
            electron.probability_ambient = self.calculate_probability_boundary(electron)
            # electron.count_ambient = self.calculate_count_boundary(electron)
            # print(electron.probability)
        if electron.collision_type == CollisionTypes.Spacecraft:
            self.calculate_probability_spacecraft(electron)
            self.calculate_count_spacecraft(electron)
            # electron.probability_photo = self.calculate_probability_spacecraft_photo(electron)

    def calculate_probability_spacecraft(self, electron: Electron):
        def get_temperature_SEEE():
            return 2 * scipy.constants.physical_constants["electron volt-kelvin relationship"][0] # TODO 
        
        def get_temperature_photo():
            return 3 * scipy.constants.physical_constants["electron volt-kelvin relationship"][0] #TODO

        def norm_distribution_speed(vec: Vector3D, temperature: float) -> float:
            # print("SEEE" , vec)
            x = np.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
            kb = consts.k
            me = consts.electron_mass
            temperature = temperature
            result = np.sqrt(2/np.pi) * np.power(me/(kb * temperature), 3/2) * (x**2) * np.exp(-me*x**2/(2*kb*temperature))
            first = np.sqrt(2/np.pi) * np.power(me/(kb * temperature), 3/2)
            second = (x**2) * np.exp(-me*x**2/(2*kb*temperature))

            result = first * second
            # print("SEE CALC", first, second, result, "Argument", -me*x**2/(2*kb*temperature))
            # print(format(result, '.60g'))
            return result

        
        electron.probability_secondary = norm_distribution_speed(electron.velocity, get_temperature_SEEE())

        if self.detect_if_sun_visible_photoelectrons(electron): 
            electron.probability_photo = norm_distribution_speed(electron.velocity, get_temperature_photo())
        

        return True        
    

    
    def calculate_count_spacecraft(self, electron: Electron):
        id:int = self.mesh.mesh.find_closest_point(electron.position)
        if id == -1: 
            raise Exception

        photodensity = self.mesh.mesh['final_photoElec_charge_density_-_step0'][id]
        secdensity = self.mesh.mesh['final_secondElec_BS_from_ambiant_electrons_charge_density_-_step0'][id]


        if electron.probability_photo is not None: 
            electron.count_photo = electron.probability_photo * photodensity / scipy.constants.elementary_charge

        if electron.probability_secondary is not None:
            electron.count_secondary = electron.probability_secondary * secdensity / scipy.constants.elementary_charge

        


        return True        
    
    def calculate_count_boundary(self, electron: Electron):
        id:int = self.e_field_mesh.find_closest_cell(electron.position)
        if id == -1: 
            raise Exception

        particle_count = 10**7 # TODO 

        if electron.probability_ambient is not None:
            result:float = electron.probability_ambient * particle_count

        electron.count_ambient = result

        return result        




    def calculate_probability_boundary(self, electron:Electron):
        def norm_distribution_speed(vec: Vector3D) -> float:
            # print("Boundary", vec)
            x = np.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
            kb = consts.k
            me = consts.electron_mass
            T = self.boundary_temperature
            return np.sqrt(2/np.pi) * np.power(me/(kb * T), 3/2) * (x**2) * np.exp(-me*x**2/(2*kb*T))
        
        return norm_distribution_speed(electron.velocity)

    def get_electric_field(self, position: Vector3D) -> Vector3D:
        id: int = self.e_field_mesh.find_closest_cell(position) #type: ignore 
        if id == -1: 
            raise Exception
        result = self.e_field_mesh["vector_electric_field"][id]
        return result
    
    def get_plasma_potential(self, position: Vector3D) -> Vector3D:
        id: int = self.plasma_potential_mesh.mesh.find_closest_point(position) #type: ignore 
        if id == -1: 
            raise Exception
        result = self.plasma_potential_mesh.mesh[self.potential_name][id]

        return result

    def get_density(self, position: Vector3D) -> float:
        # id: int = self.mesh.mesh.find_containing_cell(position) #type: ignore 
        id: int = self.mesh.mesh.find_closest_point(position)
        if id == -1: 
            raise Exception
            
        result = self.mesh.mesh[self.density_name][id]
        return result
    
    def get_gradient_of_density(self, position: Vector3D) -> Vector3D:
        # id: int = self.charge_density_gradient.find_containing_cell(position) #type: ignore 
        id: int = self.charge_density_gradient.find_closest_point(position)
        if id == -1: 
            raise Exception
        result = self.charge_density_gradient["gradient"][id]
        return result


    def check_boundary_collision(self, electron :Electron) -> bool:
        if np.linalg.norm(electron.position) > 15:
            return True
        return False

    def move_backwards(self, electron: Electron, dt:float):

        E: Vector3D = self.get_electric_field(electron.position)

        # electron.potential_history.append(self.get_plasma_potential(electron.position))
        # electron.energy_history.append(electron.get_energy())
        # electron.dt_history.append(dt)


        electron.number_steps_taken += 1
        if electron.number_steps_taken < 50:
            dt *= 0.7

        if electron.number_steps_taken < 3:
            rk_scheme(electron, -dt, E, self.get_electric_field)
            return
        
        if electron.number_steps_taken < 50:
            dt *= 0.7
        if electron.number_steps_taken > 100:
            dt *= 3
        if electron.number_steps_taken > 200:
            dt *= 4
        if electron.number_steps_taken > 400:
            dt *= 6
        if electron.number_steps_taken > 800:
            dt *= 8

        if self.backtracking_type == BacktrackingTypes.Euler:
            # Euler scheme
            euler_scheme(electron, -dt, E)

        elif self.backtracking_type == BacktrackingTypes.RK:
            # Runge-Kutta 2nd order (midpoint method)
            rk_scheme(electron, -dt, E, self.get_electric_field)


        elif self.backtracking_type == BacktrackingTypes.Boris:
            # Boris push without magnetic field
            boris_scheme(electron, -dt, E)

        return
            

    def detect_collision_SC(self, electron: Electron):
        spacecraft_mesh = self.simulation.results.extracted_data_fields.spacecraft_face.mesh

        collided_faces_SC: list[Any] = spacecraft_mesh.find_cells_intersecting_line(electron.previous_position, #type: ignore
                                                                                   electron.position) 
        
        return collided_faces_SC 
    
    def detect_if_sun_visible_photoelectrons(self, electron: Electron):
        spacecraft_mesh = self.simulation.results.extracted_data_fields.spacecraft_face.mesh

        sun_direction = [-1, 0, 0] # TODO

        ray = [electron.position[i] + 30 * sun_direction[i] for i in range(3) ]

        collided_faces_SC: list[Any] = spacecraft_mesh.find_cells_intersecting_line(electron.position, #type: ignore
                                                                                   ray) 
         
        # print(collided_faces_SC)

        # electron.position_history.append(ray)

        direction_check = [sun_direction[i] * (electron.previous_position[i] - electron.position[i])  for i in range(3)]

        direction_check_result = direction_check[0] + direction_check[1] + direction_check[2]
        
        return (len(collided_faces_SC) == 0)  and direction_check_result < 0
    

    def generate_electron_vector(self, direction: Vector3D, angle: Vector2D):
        origin = (angle[1], angle[0])

        position = self.position + self.radius*direction 


        # origin = self._map_point_into_2d_plane(self._get_starting_position_direction(x,y))

        energy = self.energy *  scipy.constants.eV

        velocity = self._get_starting_velocity(position, energy)
        # print("velocity", velocity)


        result = Electron(origin=origin, position=position, 
                          previous_position=position, velocity=velocity, previous_velocity=velocity, starting_energy=self.energy)

        self.particles.append(result)
        return result
    

    def generate_electron_at_angle(self, x: float,y: float):
        
        

        origin = np.array([x,y])

        position = self._get_starting_position_angle(x,y)


        # origin = self._map_point_into_2d_plane(self._get_starting_position_direction(x,y))

        energy = self.energy *  scipy.constants.eV

        velocity = self._get_starting_velocity(position, energy)


        result = Electron(origin=origin, position=position, 
                          previous_position=position, velocity=velocity, previous_velocity=velocity, starting_energy=self.energy)

        self.particles.append(result)
        return result

    def generate_electron(self) -> Electron:
        
        while True:
           point_on_a_sphere = self._random_point_unit_sphere()
           if np.dot(point_on_a_sphere, self.orientation) > np.cos(self.acceptance_angle_phi):
               break
        
        
        origin = self._map_point_into_2d_plane(point_on_a_sphere)

        position = self._get_starting_position(point_on_a_sphere)

        energy = self.energy *  scipy.constants.eV

        velocity = self._get_starting_velocity(position, energy)


        result = Electron(origin=origin, position=position, 
                          previous_position=position, velocity=velocity, previous_velocity=velocity, starting_energy=self.energy)

        self.particles.append(result)
        return result
    
    def _get_starting_position(self, direction: Vector3D) -> Vector3D:
        result = self.position + direction*self.radius 
        return result

    def _get_starting_position_angle(self, theta: float ,phi: float) -> Vector3D:
        direction = np.array([np.sin(theta)*np.cos(phi), 
                              np.sin(theta)*np.sin(phi), 
                              np.cos(theta)])


        result = self.position + direction*self.radius 
        return result
    
    def _get_starting_position_direction(self, theta: float ,phi: float) -> Vector3D:
        direction = np.array([np.sin(theta)*np.cos(phi), 
                              np.sin(theta)*np.sin(phi), 
                              np.cos(theta)])

        return direction


    def _convert_origin_to_polar(self, point:Vector3D) -> tuple[float, float]:
        return utils.cartesian_to_polar(point=point, direction=self.orientation, up=self.updirection)
    
    def _map_point_into_2d_plane(self, input:Vector3D) -> Vector2D:
        # return np.array((input[1],input[2]))
        return np.array(self._convert_origin_to_polar(input))
    
    def _random_point_unit_sphere(self) -> Vector3D:
        x = np.random.normal()
        y = np.random.normal()
        z = np.random.normal()

        length: float = np.sqrt(x*x + y*y + z*z)
        return np.array([x,y,z])/length

    def _get_starting_velocity(self, position:Vector3D, energy: float) -> Vector3D:
        

        speed = np.sqrt(2 * energy/ Electron.MASS)
        direction =   self.position - position
        direction = direction / np.linalg.norm(direction)

        direction *= speed



        return direction


    def accumulate_collision(self, electron: Electron):

        self.result_accumulator.add_particle(electron)



def euler_scheme(electron: Electron, dt:float, E:Vector3D):
        new_velocity  = electron.velocity + ((dt) * (-electron.CHARGE/electron.MASS) * E)
        new_position  = electron.position + ((dt) * electron.velocity)

        electron.previous_velocity = electron.velocity
        electron.previous_position = electron.position

        electron.velocity = new_velocity
        electron.position = new_position

def rk_scheme(electron: Electron, dt:float, E:Vector3D, func_for_efield):
        k1_v = (-electron.CHARGE / electron.MASS) * E
        k1_x = electron.velocity

        mid_position = electron.position + (0.5 * dt) * k1_x
        mid_velocity = electron.velocity + (0.5 * dt) * k1_v

        E_mid = func_for_efield(mid_position)
        k2_v = (-electron.CHARGE / electron.MASS) * E_mid
        k2_x = mid_velocity

        new_velocity = electron.velocity + (dt) * k2_v
        new_position = electron.position + (dt) * k2_x

        electron.previous_velocity = electron.velocity
        electron.previous_position = electron.position

        electron.velocity = new_velocity
        electron.position = new_position

def boris_scheme(electron: Electron, dt:float, E:Vector3D):
        # Boris push without magnetic field
        qmdt2 = (-electron.CHARGE / electron.MASS) * (dt / 2.0)

        # Half acceleration by E
        v_minus = electron.velocity + qmdt2 * E

        # B would be here 

        # Half acceleration again
        v_plus = v_minus + qmdt2 * E

        new_velocity = v_plus
        new_position = electron.position + (dt) * v_plus

        electron.previous_velocity = electron.velocity
        electron.previous_position = electron.position

        electron.velocity = new_velocity
        electron.position = new_position