"""Default configuration values for the flocking simulation."""

FLOCKING_DEFAULTS = {
    "cohesion": 0.5,
    "alignment": 2.2,
    "separation": 1.3,
    "wall_avoidance": 3.6,
    "leader_bias": 4.3,
    "separation_radius": 25.0,
    "perception_radius": 100.0
}

PREDATOR_PREY_CONFIG = {
    "num_predators": 5,
    "num_prey": 20,
    "predator_speed": 1.2,  # Faster than prey
    "prey_speed": 1.0,
    "catch_distance": 5.0,
    "predator_vision_range": 100,
    "prey_vision_range": 120,  # Prey might see further
    "simulation_steps": 1000,
    "world_size": (800, 800),
    "cell_size": 50  # For spatial partitioning
} 