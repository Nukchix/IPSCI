from enum import Enum
from dataclasses import dataclass, field


S_WIDTH = 1200.0
S_HEIGHT = 800.0
S_DEPTH = 800.0


class SpaceDimension(str, Enum):
    TWO_D = "2d"
    THREE_D = "3d"


class GravityMode(str, Enum):
    NONE = "none"
    UNIFORM_GRAVITY = "uniform_gravity"
    N_BODY_GRAVITY = "n_body_gravity"

@dataclass
class Particle:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass: float = 1.0
    radius: float = 10.0
    charge: float = 0.0
    color: str = "#777777"

    @property
    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * (
            self.vx * self.vx + self.vy * self.vy + self.vz * self.vz
        )

class PhysicsWorld:
    def __init__(self) -> None:
        self.width = S_WIDTH
        self.height = S_HEIGHT
        self.depth = S_DEPTH
        self.dimension = SpaceDimension.THREE_D
        self.gravity_mode = GravityMode.NONE
        self.electric_enabled = False
        self.magnetic_enabled = False
        self.walls_enabled = True
        self.collisions_enabled = True
        self.time = 0.0
        self.collision_checks = 0
        self.particles: list[Particle] = []

    def total_kinetic_energy(self) -> float:
        return sum(particle.kinetic_energy for particle in self.particles)