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

Vector3D: TypeAlias = np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]
Vector2D: TypeAlias = np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]



DEFAULT_PROJECTION_DIRECTION: Vector3D = np.array([1,0,0])

class CollisionTypes(Enum):
    No_collision = 0
    Spacecraft = 1 
    Boundary = 2 
    Too_many_steps = 3 

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

    probability: float|None = None


class ResultAccumulator():
    def add_particle(self, electron: Electron):
        pass

    def plot(self):
        pass

class ScatterResultAccumulator(ResultAccumulator):
    def __init__(self) -> None:
        super().__init__()
        self.particle_origins_hit_spacecraft: list[Tuple[Vector2D, float]] = []
        self.particle_origins_miss_spacecraft: list[Tuple[Vector2D, float]]  = []
        self.particle_origins_unknown: list[Tuple[Vector2D, float]]  = []

    def add_particle(self, electron: Electron):
        x = electron.origin

        alpha = electron.probability if electron.probability is not None else 1


        if electron.collision_type == CollisionTypes.Boundary:
            self.particle_origins_miss_spacecraft.append((x, alpha))
        elif electron.collision_type == CollisionTypes.Spacecraft:
            self.particle_origins_hit_spacecraft.append((x, alpha))
        else:
            self.particle_origins_unknown.append((x, alpha))

    def plot(self):
        
        # find min max
        minimum = 10
        maximum = 0
        for i in self.particle_origins_miss_spacecraft:
            pos = i[0]
            alpha = i[1]

            minimum = min(minimum, alpha)
            maximum = max(maximum, alpha)


        xx = []
        yy = []
        cc = []
        for i in self.particle_origins_miss_spacecraft:
            pos = i[0]
            alpha = i[1]

            print(alpha/maximum)

            xx.append(pos[0])
            yy.append(pos[1])
            cc.append(alpha)            
            # plt.scatter(pos[0], pos[1], c=alpha/maximum, cmap="inferno")
        plt.scatter(xx,yy,c=cc, cmap="viridis")
        plt.colorbar()


        hit_x = []
        hit_y = []
        for i in self.particle_origins_hit_spacecraft:
            pos = i[0]
            alpha = i[1]

            hit_x.append(pos[0])
            hit_y.append(pos[1])

            plt.scatter(pos[0], pos[1], c="red", alpha=alpha)

        for i in self.particle_origins_unknown:
            pos = i[0]
            alpha = i[1]
            plt.scatter(pos[0], pos[1], c="black", alpha=alpha)

        data = {'X_position': xx,
         'Y_position': yy,
         'value': cc}
        df = pd.DataFrame.from_dict(data)
        df.to_csv('boundary.csv')


        data = {'X_position': hit_x,
         'Y_position': hit_y}
        df = pd.DataFrame.from_dict(data)
        df.to_csv('spacecraft.csv')

        plt.show()



class ElectronDetector:
    def __init__(self, data: simulation.Simulation) -> None:
        self.particles: list[Electron] = []
        self.position: Vector3D = np.array([3.4466, 0,-0.135])
        self.orientation: Vector3D = np.array([0, 0, -1])
        self.updirection: Vector3D = np.array([1, 0 ,0])
        self.radius: float = 0.15
        self.acceptance_angle: float = np.pi

        ## 2, 14, 50 
        self.max_energy : float = 14
        self.min_energy : float = 14

        self.time: float
        self.dt: float = 12 / (18755372 * 1)

        self.number_of_samples_y: int = 20
        self.number_of_samples_x: int = 10
        
        self.number_of_steps: int = 40


        self.simulation: simulation.Simulation = data

        self.result_accumulator : ResultAccumulator = ScatterResultAccumulator()

        self.boundary_temperature = 103000 #for 8.9 eV

        # Settings for random drawing
        self.monte_carlo = True
        self.number_of_samples: int = 600
        

        # normalize orientation vectors
        self.orientation = utils.normalize(self.orientation)
        self.updirection = utils.normalize(self.updirection)

        self.mesh = utils.generate_efield_vector_property(self.simulation) # Contains "vector_electric_field"


    def get_trajectories(self):
        result: list[list[Vector3D]] = []
        for particle in self.particles:
            result.append(particle.position_history)
        return result


    def get_typed_trajectories(self):
        result: list[Tuple[list[Vector3D], CollisionTypes]] = []
        for particle in self.particles:
            result.append(( particle.position_history, particle.collision_type))
        return result


    def backtrack_monte_carlo(self):
        for _ in range(self.number_of_samples):
            electron = self.generate_electron()

            self.backtrack_one_electron(electron)

            if electron.collision_type != CollisionTypes.No_collision:
                self.acumulate_colission(electron)

        return 
    
    def backtrack_grid(self):
        for x in np.linspace(0, np.pi, num=self.number_of_samples_x):
            print(x)
            for y in np.linspace(-np.pi, np.pi, num=self.number_of_samples_y): 
                electron = self.generate_electron_at_angle(x,y)

                self.backtrack_one_electron(electron)

                if electron.collision_type != CollisionTypes.No_collision:
                    self.acumulate_colission(electron)

        return 


    def backtrack(self):
        if self.monte_carlo == True:
            self.backtrack_monte_carlo()
        else:
            self.backtrack_grid()

    def backtrack_one_electron(self, electron: Electron):
            colided = False
            step = 0

            electron.position_history.append(electron.position)
            while not colided:

                step += 1 

                try:
                    self.move_backwards(electron, self.dt)
                except Exception:
                    colided = True
                    electron.collision_type = CollisionTypes.Boundary

                electron.position_history.append(electron.position)

                if self.check_boundary_collision(electron):
                    colided = True
                    electron.collision_type = CollisionTypes.Boundary
                if self.detect_colission_SC(electron):
                    colided = True
                    electron.collision_type = CollisionTypes.Spacecraft
                if step > self.number_of_steps:
                    colided = True
                    electron.collision_type = CollisionTypes.Too_many_steps
            
            self.calculate_probability(electron)


    def calculate_probability(self, electron: Electron) -> None:
        if electron.collision_type == CollisionTypes.Boundary:
            electron.probability = self.calculate_probability_boundary(electron)
            # print(electron.probability)


    def calculate_probability_boundary(self, electron:Electron):
        def norm_distribution_speed(vec: Vector3D) -> float:
            x = np.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
            kb = consts.k
            me = consts.electron_mass
            T = self.boundary_temperature
            return np.sqrt(2/np.pi) * np.power(me/(kb * T), 3/2) * (x**2) * np.exp(-me*x**2/(2*kb*T))
        
        return norm_distribution_speed(electron.velocity)

    def get_electric_field(self, position: Vector3D) -> Vector3D:
        id: int = self.mesh.find_containing_cell(position) #type: ignore 
        if id == -1: 
            raise Exception
        result = self.mesh["vector_electric_field"][id]
        return result

    

    def check_boundary_collision(self, electron :Electron) -> bool:
        if np.linalg.norm(electron.position) > 15:
            return True
        return False

    def move_backwards(self, electron: Electron, dt:float):

        E: Vector3D = self.get_electric_field(electron.position)
        
        new_velocity  = electron.velocity + -dt * (-electron.CHARGE/electron.MASS) * E
        new_position  = electron.position + -dt * new_velocity

        electron.previous_velocity = electron.velocity
        electron.previous_position = electron.position

        electron.velocity = new_velocity
        electron.position = new_position

        return

    def detect_colission_SC(self, electron: Electron) -> bool:
        spacecraft_mesh = self.simulation.results.extracted_data_fields.spacecraft_face.mesh

        colided_faces_SC: list[Any] = spacecraft_mesh.find_cells_intersecting_line(electron.previous_position, #type: ignore
                                                                                   electron.position) 
        return len(colided_faces_SC) > 0 
    

    def generate_electron_at_angle(self, x: float,y: float):
        
        

        # origin = np.array([x,y])

        position = self._get_starting_position_angle(x,y)


        origin = self._map_point_into_2d_plane(self._get_starting_position_direction(x,y))

        energy = random.uniform(self.min_energy,self.max_energy) *  scipy.constants.eV

        velocity = self._get_starting_velocity(position, energy)


        result = Electron(origin=origin, position=position, 
                          previous_position=position, velocity=velocity, previous_velocity=velocity, starting_energy=energy)

        self.particles.append(result)
        return result

    def generate_electron(self) -> Electron:
        
        while True:
           point_on_a_sphere = self._random_point_unit_sphere()
           if np.dot(point_on_a_sphere, self.orientation) > np.cos(self.acceptance_angle):
               break
        
        
        origin = self._map_point_into_2d_plane(point_on_a_sphere)

        position = self._get_starting_position(point_on_a_sphere)

        energy = random.uniform(self.min_energy,self.max_energy) *  scipy.constants.eV

        velocity = self._get_starting_velocity(position, energy)


        result = Electron(origin=origin, position=position, 
                          previous_position=position, velocity=velocity, previous_velocity=velocity, starting_energy=energy)

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
        direction =  self.position - position
        direction /= np.linalg.norm(direction)
        direction *= speed



        return direction


    def acumulate_colission(self, electron: Electron):
        self.result_accumulator.add_particle(electron)
