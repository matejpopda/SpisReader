import simulation
from typing import Tuple, Any, TypeAlias, ClassVar, Literal
from dataclasses import dataclass
import scipy.constants # type: ignore
import numpy as np
import random
from enum import Enum
import utils
import matplotlib.pyplot as plt


Vector3D: TypeAlias = np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]
Vector2D: TypeAlias = np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]



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


class ResultAccumulator():
    def add_particle(self, electron: Electron):
        pass

    def plot(self):
        pass

class ScatterResultAccumulator(ResultAccumulator):
    def __init__(self) -> None:
        super().__init__()
        self.particle_origins_hit_spacecraft: list[Vector2D] = []
        self.particle_origins_miss_spacecraft: list[Vector2D] = []
        self.particle_origins_unknown: list[Vector2D] = []

    def add_particle(self, electron: Electron):
        x = electron.origin

        if electron.collision_type == CollisionTypes.Boundary:
            self.particle_origins_miss_spacecraft.append(x)
        if electron.collision_type == CollisionTypes.Spacecraft:
            self.particle_origins_hit_spacecraft.append(x)
        else:
            self.particle_origins_unknown.append(x)

    def plot(self):
        for i in self.particle_origins_miss_spacecraft:
            plt.scatter(i[0], i[1], c="blue")

        for i in self.particle_origins_hit_spacecraft:
            plt.scatter(i[0], i[1], c="red")

        # for i in self.particle_origins_unknown:
        #     plt.scatter(i[0], i[1], c="green")
        plt.show()



class ElectronDetector:
    def __init__(self, data: simulation.Simulation) -> None:
        self.particles: list[Any] = []
        self.position: Vector3D = np.array([-10, 0, 0])
        self.orientation: Vector3D
        self.updirection: Vector3D
        self.radius: float = 0.1
        self.cone_:float 
        self.acceptance_angle: Vector2D

        self.time: float
        self.dt: float = 12 / 18755372 

        self.number_of_samples: int = 500
        self.number_of_steps: int = 15


        self.simulation: simulation.Simulation = data

        self.result_accumulator : ResultAccumulator = ScatterResultAccumulator()

        self.mesh = utils.generate_efield_vector_property(self.simulation) # Contains "vector_electric_field"

    
    def backtrack(self):
        for _ in range(self.number_of_samples):
            electron = self.generate_electron()

            self.backtrack_one_electron(electron)

            if electron.collision_type != CollisionTypes.No_collision:
                self.acumulate_colission(electron)

        return 

    def backtrack_one_electron(self, electron: Electron):
            colided = False
            step = 0

            while not colided:
                step += 1 
                self.move_backwards(electron, self.dt)


                if self.check_boundary_collision(electron):
                    colided = True
                    electron.collision_type = CollisionTypes.Boundary
                if self.detect_colission_SC(electron):
                    colided = True
                    electron.collision_type = CollisionTypes.Spacecraft
                if step > self.number_of_steps:
                    colided = True
                    electron.collision_type = CollisionTypes.Too_many_steps


    def get_electric_field(self, position: Vector3D) -> Vector3D:
        id: int = self.mesh.find_containing_cell(position)
        result = self.mesh["vector_electric_field"][id]

        # print(result)

        return result
        self.mesh["vector_electric_field"]
        return np.array((0,0,0))
        self.mesh["vector_electric_field"]
    

    def check_boundary_collision(self, electron :Electron) -> bool:
        if np.linalg.norm(electron.position) > 15:
            return True
        return False

    def move_backwards(self, electron: Electron, dt:float):

        E: Vector3D = self.get_electric_field(electron.position)
        
        new_velocity  = electron.velocity + -dt * (electron.CHARGE/electron.MASS) * E
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
    

    def generate_electron(self) -> Electron:
        point_on_a_sphere = self._random_point_unit_sphere()
        
        origin = self._map_point_into_2d_plane(point_on_a_sphere)

        position = self._get_starting_position(point_on_a_sphere)

        energy = 1000 *  scipy.constants.eV

        velocity = self._get_starting_velocity(position, energy)


        result = Electron(origin=origin, position=position, 
                          previous_position=position, velocity=velocity, previous_velocity=velocity, starting_energy=energy)

        return result
    
    def _get_starting_position(self, origin: Vector3D) -> Vector3D:
        result = self.position + origin*self.radius 
        return result

    def _map_point_into_2d_plane(self, input:Vector3D) -> Vector2D:
        return np.array((input[1],input[2]))

    def _random_point_unit_sphere(self) -> Vector3D:
        x = np.random.normal()
        y = np.random.normal()
        z = np.random.normal()

        length: float = np.sqrt(x*x + y*y + z*z)
        return np.array([x,y,z])/length

    def _get_starting_velocity(self, position:Vector3D, energy: float) -> Vector3D:
        

        speed = np.sqrt(2 * energy/ Electron.MASS)
        direction = position - self.position
        direction /= np.linalg.norm(direction)
        direction *= speed



        return direction


    def acumulate_colission(self, electron: Electron):
        self.result_accumulator.add_particle(electron)
