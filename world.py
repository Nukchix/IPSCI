import math
from enum import Enum
from collections import deque
from dataclasses import dataclass, field


S_WIDTH = 1200.0
S_HEIGHT = 800.0
S_DEPTH = 800.0
GRAVITATIONAL_CONSTANT_SI = 6.67430e-11
COULOMB_CONSTANT_SI = 8.9875517923e9
INTERACTION_SOFTENING_SI = 1e-6
SPEED_OF_LIGHT = 299_792_458.0
VACUUM_PERMEABILITY = 4 * math.pi * 1e-7


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
    trail_enabled: bool = False
    trail: deque[tuple[float, ...]] = field(default_factory=deque)

    @property
    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * (
            self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

class PhysicsWorld:
    MAX_INTEGRATION_STEP = 1.0 / 120.0
    MAX_TRAIL_POINTS = 400
    ADAPTIVE_SAFETY = 0.04
    
    def __init__(self) -> None:
        self.width = S_WIDTH
        self.height = S_HEIGHT
        self.depth = S_DEPTH
        self.dimension = SpaceDimension.THREE_D
        self.gravity_mode = GravityMode.NONE
        self.gravity_x = 0.0
        self.gravity_y = 0.0
        self.gravity_z = 0.0
        self.electric_enabled = False
        self.magnetic_enabled = False
        self.walls_enabled = True
        self.collisions_enabled = True
        self.time = 0.0
        self.collision_checks = 0
        self.particles: list[Particle] = []
        self.gravitational_constant = GRAVITATIONAL_CONSTANT_SI
        self.gravitational_softening = INTERACTION_SOFTENING_SI
        self.coulomb_constant = COULOMB_CONSTANT_SI
        self.electric_softening = INTERACTION_SOFTENING_SI
        self.viscosity = 0.0
        self.trail_duration = 10.0

    def step(self, dt: float) -> None:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Integration step must be a positive finite number")
        remaining = dt
        substeps = 0
        while remaining > max(dt * 1e-14, 1e-15):
            step_dt = min(
                remaining,
                self.MAX_INTEGRATION_STEP,
                self._stable_timestep(remaining), )
            if step_dt <= 0 or not math.isfinite(step_dt):
                raise ArithmeticError("Could not determine a stable integration step")
            self._step_once(step_dt)
            remaining -= step_dt
            substeps += 1
            if substeps >= 100_000 and remaining > max(dt * 1e-14, 1e-15):
                raise ArithmeticError(
                    "Adaptive integration exceeded 100,000 substeps; "
                    "reduce charges, increase masses, or slow simulation time")
        if remaining > 0:
            self._step_once(remaining)

    def _stable_timestep(self, remaining: float) -> float:
        particles = self.particles
        if len(particles) < 1:
            return min(remaining, self.MAX_INTEGRATION_STEP)
        ax, ay, az = self._accelerations()
        timestep = self.MAX_INTEGRATION_STEP
        safety = self.ADAPTIVE_SAFETY

        for first_index, first in enumerate(particles):
            for second_index in range(first_index + 1, len(particles)):
                second = particles[second_index]
                dx = second.x - first.x
                dy = second.y - first.y
                dz = (
                    second.z - first.z
                    if self.dimension == SpaceDimension.THREE_D
                    else 0.0)
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                relative_speed = math.sqrt(
                    (second.vx - first.vx) ** 2
                    + (second.vy - first.vy) ** 2
                    + (
                        (second.vz - first.vz) ** 2
                        if self.dimension == SpaceDimension.THREE_D
                        else 0.0) )
                active_force = (
                    (self.electric_enabled and first.charge != 0 and second.charge != 0)
                    or self.gravity_mode == GravityMode.N_BODY_GRAVITY
                    or (
                        self.magnetic_enabled
                        and first.charge != 0
                        and second.charge != 0) )
                if active_force:
                    force_softening = max(
                        self.gravitational_softening
                        if self.gravity_mode == GravityMode.N_BODY_GRAVITY
                        else 0.0,
                        self.electric_softening
                        if self.electric_enabled or self.magnetic_enabled
                        else 0.0)
                    length_scale = max(distance, force_softening)
                    relative_acceleration = math.sqrt(
                        (ax[second_index] - ax[first_index]) ** 2
                        + (ay[second_index] - ay[first_index]) ** 2
                        + (az[second_index] - az[first_index]) ** 2)
                    if relative_acceleration > 0:
                        timestep = min(
                            timestep,
                            safety * math.sqrt(length_scale / relative_acceleration) )
                    if relative_speed > 0:
                        timestep = min(
                            timestep,
                            safety * length_scale / relative_speed)
                if self.collisions_enabled:
                    gap = distance - first.radius - second.radius
                    if gap <= 0 or distance <= 0:
                        continue
                    relative_velocity = (
                        second.vx - first.vx,
                        second.vy - first.vy,
                        second.vz - first.vz
                        if self.dimension == SpaceDimension.THREE_D
                        else 0.0)
                    direction = (dx / distance, dy / distance, dz / distance)
                    closing_speed = -sum(
                        displacement * velocity
                        for displacement, velocity in zip(
                            direction, relative_velocity) )
                    if closing_speed > 0:
                        relative_acceleration = (
                            ax[second_index] - ax[first_index],
                            ay[second_index] - ay[first_index],
                            az[second_index] - az[first_index])
                        closing_acceleration = -sum(
                            direction[axis] * relative_acceleration[axis]
                            for axis in range(3) )
                        discriminant = (
                            closing_speed**2
                            + 2.0 * closing_acceleration * gap)
                        if discriminant >= 0:
                            denominator = closing_speed + math.sqrt(discriminant)
                            if denominator > 0:
                                time_to_collision = 2.0 * gap / denominator
                                timestep = min(
                                    timestep,
                                    time_to_collision * (1.0 + 1e-8) )
        return min(remaining, max(timestep, 1e-15))

    def _step_once(self, dt: float) -> None:
        particles = self.particles
        count = len(particles)
        if not count:
            self.time += dt
            return

        initial_ax, initial_ay, initial_az = self._accelerations()
        for index, particle in enumerate(particles):
            particle.vx += 0.5 * initial_ax[index] * dt
            particle.vy += 0.5 * initial_ay[index] * dt
            particle.vz += 0.5 * initial_az[index] * dt
            if self.viscosity > 0:
                damping = math.exp(-self.viscosity * dt / particle.mass)
                particle.vx *= damping
                particle.vy *= damping
                particle.vz *= damping
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            if self.dimension == SpaceDimension.THREE_D:
                particle.z += particle.vz * dt
            else:
                particle.z = self.depth / 2
                particle.vz = 0.0
            if self.walls_enabled:
                self._resolve_walls(particle)

        if self.collisions_enabled:
            self._resolve_collisions()
        final_ax, final_ay, final_az = self._accelerations()
        for index, particle in enumerate(particles):
            particle.vx += 0.5 * final_ax[index] * dt
            particle.vy += 0.5 * final_ay[index] * dt
            particle.vz += 0.5 * final_az[index] * dt
            if particle.trail_enabled:
                trail_time = self.time + dt
                sample_interval = max(
                    self.MAX_INTEGRATION_STEP,
                    self.trail_duration / self.MAX_TRAIL_POINTS)
                if (
                    not particle.trail
                    or trail_time - particle.trail[-1][0] >= sample_interval ):
                    particle.trail.append(
                        (trail_time, particle.x, particle.y, particle.z) )
                cutoff = trail_time - self.trail_duration
                while particle.trail and particle.trail[0][0] < cutoff:
                    particle.trail.popleft()
                while len(particle.trail) > self.MAX_TRAIL_POINTS:
                    particle.trail.popleft()
        if not self.collisions_enabled:
            self.collision_checks = 0
        self.time += dt

    def _accelerations(self) -> tuple[list[float], list[float], list[float]]:
        particles = self.particles
        count = len(particles)
        if self.gravity_mode == GravityMode.UNIFORM_GRAVITY:
            ax = [self.gravity_x] * count
            ay = [self.gravity_y] * count
            az = (
                [self.gravity_z] * count
                if self.dimension == SpaceDimension.THREE_D
                else [0.0] * count)
        else:
            ax = [0.0] * count
            ay = [0.0] * count
            az = [0.0] * count

        if self.electric_enabled and count > 1:
            for i, first in enumerate(particles):
                if first.charge == 0:
                    continue
                for j in range(i + 1, count):
                    second = particles[j]
                    if second.charge == 0:
                        continue
                    dx = second.x - first.x
                    dy = second.y - first.y
                    dz = (
                        second.z - first.z
                        if self.dimension == SpaceDimension.THREE_D
                        else 0.0)
                    distance_sq = dx * dx + dy * dy + dz * dz + self.electric_softening**2
                    scale = (
                        self.coulomb_constant
                        * first.charge
                        * second.charge
                        / (distance_sq * math.sqrt(distance_sq)) )
                    ax[i] -= scale * dx / first.mass
                    ay[i] -= scale * dy / first.mass
                    az[i] -= scale * dz / first.mass
                    ax[j] += scale * dx / second.mass
                    ay[j] += scale * dy / second.mass
                    az[j] += scale * dz / second.mass
        if self.gravity_mode == GravityMode.N_BODY_GRAVITY and count > 1:
            for i, first in enumerate(particles):
                for j in range(i + 1, count):
                    second = particles[j]
                    dx = second.x - first.x
                    dy = second.y - first.y
                    dz = (
                        second.z - first.z
                        if self.dimension == SpaceDimension.THREE_D
                        else 0.0)
                    distance_sq = (
                        dx * dx + dy * dy + dz * dz + self.gravitational_softening**2)
                    scale = self.gravitational_constant / (
                        distance_sq * math.sqrt(distance_sq) )
                    ax[i] += scale * second.mass * dx
                    ay[i] += scale * second.mass * dy
                    az[i] += scale * second.mass * dz
                    ax[j] -= scale * first.mass * dx
                    ay[j] -= scale * first.mass * dy
                    az[j] -= scale * first.mass * dz
        if self.magnetic_enabled and count > 1:
            source_accelerations = list(zip(ax, ay, az))
            for target_index, target in enumerate(particles):
                if target.charge == 0:
                    continue
                target_velocity = (
                    target.vx,
                    target.vy,
                    target.vz
                    if self.dimension == SpaceDimension.THREE_D
                    else 0.0)
                for source_index, source in enumerate(particles):
                    if source_index == target_index or source.charge == 0:
                        continue
                    displacement = (
                        target.x - source.x,
                        target.y - source.y,
                        target.z - source.z
                        if self.dimension == SpaceDimension.THREE_D
                        else 0.0)
                    source_acceleration = source_accelerations[source_index]
                    magnetic_field = self._magnetic_field_from_charge(
                        source.charge,
                        (
                            source.vx,
                            source.vy,
                            source.vz
                            if self.dimension == SpaceDimension.THREE_D
                            else 0.0),
                        source_acceleration,
                        displacement)
                    force_per_charge = self._cross(
                        target_velocity, magnetic_field)
                    ax[target_index] += (
                        target.charge * force_per_charge[0] / target.mass)
                    ay[target_index] += (
                        target.charge * force_per_charge[1] / target.mass)
                    if self.dimension == SpaceDimension.THREE_D:
                        az[target_index] += (
                            target.charge * force_per_charge[2] / target.mass)
        return ax, ay, az

    @staticmethod
    def _cross(
        first: tuple[float, float, float],
        second: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0])

    @classmethod
    def _magnetic_field_from_charge(
        cls,
        charge: float,
        velocity: tuple[float, float, float],
        acceleration: tuple[float, float, float],
        displacement: tuple[float, float, float]) -> tuple[float, float, float]:
        distance_sq = (
            sum(component * component for component in displacement)
            + INTERACTION_SOFTENING_SI**2)
        if charge == 0:
            return (0.0, 0.0, 0.0)
        distance = math.sqrt(distance_sq)
        direction = tuple(component / distance for component in displacement)
        velocity_field = cls._cross(velocity, displacement)
        acceleration_field = cls._cross(direction, acceleration)
        coefficient = VACUUM_PERMEABILITY / (4 * math.pi) * charge
        return tuple(
            coefficient
            * (
                velocity_field[axis] / distance**3
                - acceleration_field[axis] / (SPEED_OF_LIGHT * distance) )
            for axis in range(3) )

    def _resolve_walls(self, particle: Particle) -> None:
        self._reflect_at_walls(particle, "x", "vx", self.width)
        self._reflect_at_walls(particle, "y", "vy", self.height)
        if self.dimension == SpaceDimension.THREE_D:
            self._reflect_at_walls(particle, "z", "vz", self.depth)
        else:
            particle.z = self.depth / 2
            particle.vz = 0.0

    @staticmethod
    def _reflect_at_walls(
        particle: Particle,
        position_name: str,
        velocity_name: str,
        extent: float) -> None:
        radius = particle.radius
        span = extent - 2 * radius
        if span <= 0:
            setattr(particle, position_name, extent / 2)
            setattr(particle, velocity_name, 0.0)
            return
        raw_position = getattr(particle, position_name) - radius
        phase = raw_position % (2 * span)
        if phase <= span:
            position = radius + phase
            direction = 1
        else:
            position = radius + 2 * span - phase
            direction = -1
        setattr(particle, position_name, position)
        setattr(
            particle,
            velocity_name,
            getattr(particle, velocity_name) * direction)

    def _resolve_collisions(self) -> None:
        if len(self.particles) < 2:
            self.collision_checks = 0
            return

        cell_size = max(2.0 * max(p.radius for p in self.particles), 1.0)
        buckets: dict[tuple[int, ...], list[int]] = {}
        for index, particle in enumerate(self.particles):
            key = (int(particle.x // cell_size), int(particle.y // cell_size))
            if self.dimension == SpaceDimension.THREE_D:
                key += (int(particle.z // cell_size),)
            buckets.setdefault(key, []).append(index)

        offsets = (-1, 0, 1)
        checks = 0
        for cell, indices in buckets.items():
            candidates: list[int] = []
            if self.dimension == SpaceDimension.THREE_D:
                for offset_x in offsets:
                    for offset_y in offsets:
                        for offset_z in offsets:
                            neighbor = (
                                cell[0] + offset_x,
                                cell[1] + offset_y,
                                cell[2] + offset_z)
                            candidates.extend(buckets.get(neighbor, ()))
            else:
                for offset_x in offsets:
                    for offset_y in offsets:
                        neighbor = (cell[0] + offset_x, cell[1] + offset_y)
                        candidates.extend(buckets.get(neighbor, ()))
            for first_index in indices:
                for second_index in candidates:
                    if second_index <= first_index:
                        continue
                    checks += 1
                    first = self.particles[first_index]
                    second = self.particles[second_index]
                    dx = second.x - first.x
                    dy = second.y - first.y
                    dz = (
                        second.z - first.z
                        if self.dimension == SpaceDimension.THREE_D
                        else 0.0)
                    minimum_distance = first.radius + second.radius
                    distance_sq = dx * dx + dy * dy + dz * dz
                    if distance_sq >= minimum_distance * minimum_distance:
                        continue

                    distance = math.sqrt(distance_sq) if distance_sq > 1e-12 else 1e-6
                    nx = dx / distance
                    ny = dy / distance
                    nz = dz / distance if self.dimension == SpaceDimension.THREE_D else 0.0
                    overlap = minimum_distance - distance
                    inverse_first = 1.0 / first.mass
                    inverse_second = 1.0 / second.mass
                    inverse_total = inverse_first + inverse_second
                    correction = overlap / inverse_total
                    first.x -= nx * correction * inverse_first
                    first.y -= ny * correction * inverse_first
                    first.z -= nz * correction * inverse_first
                    second.x += nx * correction * inverse_second
                    second.y += ny * correction * inverse_second
                    second.z += nz * correction * inverse_second

                    relative_velocity = (
                        (second.vx - first.vx) * nx
                        + (second.vy - first.vy) * ny
                        + (second.vz - first.vz) * nz)
                    if relative_velocity < 0:
                        impulse = -2.0 * relative_velocity / inverse_total
                        first.vx -= impulse * inverse_first * nx
                        first.vy -= impulse * inverse_first * ny
                        first.vz -= impulse * inverse_first * nz
                        second.vx += impulse * inverse_second * nx
                        second.vy += impulse * inverse_second * ny
                        second.vz += impulse * inverse_second * nz
        self.collision_checks = checks

    def total_kinetic_energy(self) -> float:
        return sum(particle.kinetic_energy for particle in self.particles)