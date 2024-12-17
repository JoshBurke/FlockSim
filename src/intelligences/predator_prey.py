import numpy as np
from typing import List, Tuple, Dict, Any
from .base import Intelligence

class PredatorIntelligence(Intelligence):
    """Intelligence for predator bots that chase and catch prey."""
    
    def __init__(self, max_speed: float = 2.0, max_force: float = 0.15, 
                 perception_radius: float = 150.0, catch_radius: float = 20.0,
                 sight_speed_ratio: float = 75.0):
        """Initialize predator intelligence.
        
        The sight_speed_ratio determines how energy is allocated between sight and speed.
        ratio = sight / speed, but both are derived from the ratio using a logarithmic function
        to prevent extreme values.
        
        Energy allocation formula:
        - Base energy = 150 (predator has more energy than prey)
        - speed = base_energy / (1 + ln(ratio))
        - sight = speed * ratio
        
        This creates a natural balance where:
        - ratio = 1: speed = 150/1.0 = 150, sight = 150
        - ratio = 10: speed = 150/3.3 ≈ 45, sight = 450
        - ratio = 100: speed = 150/5.6 ≈ 27, sight = 2700
        - ratio = 0.1: speed = 150/-1.3 ≈ 115, sight = 11.5
        """
        self.sight_speed_ratio = sight_speed_ratio
        self.base_energy = 150.0  # Total energy available
        
        # Calculate speed and sight based on ratio
        derived_speed = self.base_energy / (1 + np.log(max(sight_speed_ratio, 0.1)))
        derived_sight = derived_speed * sight_speed_ratio
        
        # Clamp values to reasonable ranges
        max_speed = min(max(derived_speed, 0.5), 4.0)
        perception_radius = min(max(derived_sight, 20.0), 400.0)
        
        super().__init__(max_speed, max_force, perception_radius)
        self.catch_radius = catch_radius
        self.color = 'darkred'  # Predator color
        self.trail_color = 'red'  # Trail color for velocity indicator
        self.weights = {
            'chase': 0.4,  # Reduced from 1.0
            'separation': 0.1,  # Reduced from 0.3
            'wall_avoidance': 0.1  # Reduced from 0.3
        }
        # Metrics for fitness calculation
        self.prey_caught = 0
        self.total_chase_distance = 0.0
        self.time_without_target = 0  # Tracks time spent without a target
        self.num_updates = 0
        
        # Constants
        self.desired_separation = 50.0  # Distance to maintain from other predators
        self.wall_margin = 30.0  # Distance to start avoiding walls
        
        # Target tracking
        self.current_target = None  # (position, frames_ago)
    
    def calculate_move(self, position: np.ndarray, velocity: np.ndarray,
                      neighbors: List[Tuple[np.ndarray, np.ndarray]], 
                      world_size: Tuple[float, float], **kwargs) -> np.ndarray:
        """Calculate next move based on visible prey and other predators.
        
        Args:
            position: Current position
            velocity: Current velocity
            neighbors: List of (position, velocity) tuples of nearby bots
            world_size: (width, height) of world
            **kwargs: Additional scenario-specific parameters including:
                     - predator_indices: List of predator bot indices
                     - caught_prey: List of caught prey indices
                     - bot_index: Index of this bot in the simulation
                     - neighbor_indices: List of indices for the neighbors list
        """
        # Get scenario parameters
        predator_indices = kwargs.get('predator_indices', [])
        caught_prey = kwargs.get('caught_prey', [])
        neighbor_indices = kwargs.get('neighbor_indices', [])  # Get list of neighbor indices
        
        # Initialize forces
        chase_force = np.zeros(2)
        separation_force = np.zeros(2)
        wall_force = np.zeros(2)
        
        # Find nearest uncaught prey
        nearest_prey_dist = float('inf')
        nearest_prey_pos = None
        
        for i, (other_pos, other_vel) in enumerate(neighbors):
            # Skip if we don't have a valid index for this neighbor
            if i >= len(neighbor_indices):
                continue
                
            neighbor_index = neighbor_indices[i]
            if neighbor_index in caught_prey:
                continue  # Ignore caught prey
            
            # Calculate distance and direction to neighbor
            to_other = other_pos - position
            distance = np.linalg.norm(to_other)
            
            if distance > 0:  # Ignore self
                if neighbor_index in predator_indices:
                    # Separation from other predators
                    if distance < self.desired_separation:
                        separation_force -= to_other / (distance + 1e-6)
                else:
                    # Potential prey found
                    if distance < nearest_prey_dist:
                        nearest_prey_dist = distance
                        nearest_prey_pos = other_pos
        
        # Update target tracking and generate chase force
        if nearest_prey_pos is not None:
            self.current_target = nearest_prey_pos
            self.time_without_target = 0
            # Calculate chase force toward nearest prey
            to_target = nearest_prey_pos - position
            chase_force = to_target / (np.linalg.norm(to_target) + 1e-6)
            # Track chase distance
            self.total_chase_distance += np.linalg.norm(velocity)
        else:
            self.time_without_target += 1
        
        # Wall avoidance
        if position[0] < self.wall_margin:  # Left wall
            wall_force[0] += 1.0
        elif position[0] > world_size[0] - self.wall_margin:  # Right wall
            wall_force[0] -= 1.0
        if position[1] < self.wall_margin:  # Bottom wall
            wall_force[1] += 1.0
        elif position[1] > world_size[1] - self.wall_margin:  # Top wall
            wall_force[1] -= 1.0
        
        # Normalize forces before applying weights
        forces = []
        for force, name in [(chase_force, 'chase'),
                          (separation_force, 'separation'),
                          (wall_force, 'wall_avoidance')]:
            if np.any(force):
                # Normalize the force
                magnitude = np.linalg.norm(force)
                if magnitude > 0:
                    force = force / magnitude
                # Apply weight and add to list
                forces.append(force * self.weights[name])
        
        # Sum all forces
        if forces:
            total_force = np.sum(forces, axis=0)
            # Normalize the total force if it exceeds max_force
            magnitude = np.linalg.norm(total_force)
            if magnitude > self.max_force:
                total_force = (total_force / magnitude) * self.max_force
        else:
            total_force = np.zeros(2)
        
        self.num_updates += 1
        return total_force
    
    def update_fitness_metrics(self, position: np.ndarray, velocity: np.ndarray,
                             neighbors: List[Tuple[np.ndarray, np.ndarray]],
                             world_size: Tuple[float, float]):
        """Update metrics used for fitness calculation."""
        # Most metrics are updated in calculate_move
        pass
    
    def calculate_fitness(self) -> float:
        """Calculate fitness score based on hunting success.
        
        Returns:
            float: Fitness score between 0 and 1
        """
        if self.num_updates == 0:
            return 0.0
            
        # Normalize metrics
        catch_score = min(1.0, self.prey_caught / 3.0)  # Cap at 3 catches
        
        # Penalize time spent without a target
        search_efficiency = max(0.0, 1.0 - (self.time_without_target / self.num_updates))
        
        # Consider chase distance (lower is better, as it means more direct catches)
        distance_score = 1.0 / (1.0 + self.total_chase_distance / self.num_updates)
        
        # Combine scores with weights
        fitness = (0.6 * catch_score +
                  0.2 * search_efficiency +
                  0.2 * distance_score)
        
        return max(0.0, min(1.0, fitness))  # Ensure between 0 and 1

class PreyIntelligence(Intelligence):
    """Intelligence for prey bots that try to evade predators."""
    
    def __init__(self, max_speed: float = 2.0, max_force: float = 0.1,
                 perception_radius: float = 100.0,
                 sight_speed_ratio: float = 50.0):
        """Initialize prey intelligence.
        
        The sight_speed_ratio determines how energy is allocated between sight and speed.
        ratio = sight / speed, but both are derived from the ratio using a logarithmic function
        to prevent extreme values.
        
        Energy allocation formula:
        - Base energy = 100 (prey has less energy than predator)
        - speed = base_energy / (1 + ln(ratio))
        - sight = speed * ratio
        
        This creates a natural balance where:
        - ratio = 1: speed = 100/1.0 = 100, sight = 100
        - ratio = 10: speed = 100/3.3 ≈ 30, sight = 300
        - ratio = 100: speed = 100/5.6 ≈ 18, sight = 1800
        - ratio = 0.1: speed = 100/-1.3 ≈ 77, sight = 7.7
        """
        self.sight_speed_ratio = sight_speed_ratio
        self.base_energy = 100.0  # Total energy available
        
        # Calculate speed and sight based on ratio
        derived_speed = self.base_energy / (1 + np.log(max(sight_speed_ratio, 0.1)))
        derived_sight = derived_speed * sight_speed_ratio
        
        # Clamp values to reasonable ranges
        max_speed = min(max(derived_speed, 0.5), 3.0)
        perception_radius = min(max(derived_sight, 20.0), 300.0)
        
        super().__init__(max_speed, max_force, perception_radius)
        self.color = 'lightblue'  # Prey color
        self.trail_color = 'blue'  # Trail color for velocity indicator
        self.weights = {
            'evade': 0.4,      # Reduced from 1.0
            'cohesion': 0.1,   # Reduced from 0.3
            'separation': 0.3,  # Reduced from 0.8
            'alignment': 0.2,   # Reduced from 0.5
            'wall_avoidance': 0.1  # Reduced from 0.3
        }
        # Metrics for fitness calculation
        self.survival_time = 0
        self.close_calls = 0  # Number of times a predator got very close
        self.total_distance = 0.0
        self.avg_group_size = 0.0  # Average size of prey group it belongs to
        self.num_updates = 0  # For averaging
        
        # Constants
        self.close_call_distance = 30.0  # Distance that counts as a close call
        self.desired_separation = 30.0  # Distance to maintain from other prey (increased from 25.0)
        self.min_cohesion_distance = 20.0  # Minimum distance for cohesion to take effect
        self.wall_margin = 50.0  # Distance to start avoiding walls
    
    def calculate_move(self, position: np.ndarray, velocity: np.ndarray,
                      neighbors: List[Tuple[np.ndarray, np.ndarray]], 
                      world_size: Tuple[float, float], **kwargs) -> np.ndarray:
        """Calculate next move based on visible predators and other prey.
        
        Args:
            position: Current position
            velocity: Current velocity
            neighbors: List of (position, velocity) tuples of nearby bots
            world_size: (width, height) of world
            **kwargs: Additional scenario-specific parameters including:
                     - predator_indices: List of predator bot indices
                     - caught_prey: List of caught prey indices
                     - bot_index: Index of this bot in the simulation
                     - neighbor_indices: List of indices for the neighbors list
        """
        # Get scenario parameters
        predator_indices = kwargs.get('predator_indices', [])
        caught_prey = kwargs.get('caught_prey', [])
        my_index = kwargs.get('bot_index', 0)  # Get bot's index from kwargs
        neighbor_indices = kwargs.get('neighbor_indices', [])  # Get list of neighbor indices
        
        # Check if this prey is caught
        if my_index in caught_prey:
            # Caught prey becomes stationary
            return -velocity  # Apply force to cancel current velocity
        
        # Initialize forces
        evade_force = np.zeros(2)
        cohesion_force = np.zeros(2)
        separation_force = np.zeros(2)
        alignment_force = np.zeros(2)
        wall_force = np.zeros(2)
        
        # Track nearby prey for group size metric and alignment
        nearby_prey = 0
        num_aligned_prey = 0
        
        # Process each neighbor
        for i, (other_pos, other_vel) in enumerate(neighbors):
            # Skip if we don't have a valid index for this neighbor
            if i >= len(neighbor_indices):
                continue
                
            neighbor_index = neighbor_indices[i]
            if neighbor_index in caught_prey:
                continue  # Ignore caught prey
                
            # Calculate distance and direction to neighbor
            to_other = other_pos - position
            distance = np.linalg.norm(to_other)
            
            if distance > 0:  # Ignore self
                if neighbor_index in predator_indices:
                    # Evade predators - stronger force when closer
                    evade_strength = 1.0 / (distance + 1e-6)
                    evade_force -= to_other * evade_strength
                    
                    # Track close calls
                    if distance < self.close_call_distance:
                        self.close_calls += 1
                else:
                    # Found another prey
                    nearby_prey += 1
                    
                    # Cohesion - move toward other prey
                    if self.min_cohesion_distance < distance < self.perception_radius:
                        cohesion_force += to_other
                    
                    # Separation - avoid getting too close
                    if distance < self.desired_separation:
                        separation_force -= to_other / (distance + 1e-6)
                    
                    # Alignment - match velocity with nearby prey
                    if distance < self.perception_radius:
                        alignment_force += other_vel
                        num_aligned_prey += 1
        
        # Average the alignment force if we found any neighbors
        if num_aligned_prey > 0:
            alignment_force = alignment_force / num_aligned_prey - velocity
        
        # Update group size metric
        if self.num_updates > 0:
            self.avg_group_size = (self.avg_group_size * self.num_updates + nearby_prey) / (self.num_updates + 1)
        else:
            self.avg_group_size = nearby_prey
        self.num_updates += 1
        
        # Wall avoidance
        if position[0] < self.wall_margin:  # Left wall
            wall_force[0] += 1.0
        elif position[0] > world_size[0] - self.wall_margin:  # Right wall
            wall_force[0] -= 1.0
        if position[1] < self.wall_margin:  # Bottom wall
            wall_force[1] += 1.0
        elif position[1] > world_size[1] - self.wall_margin:  # Top wall
            wall_force[1] -= 1.0
        
        # Normalize forces before applying weights
        forces = []
        for force, name in [(evade_force, 'evade'), 
                          (cohesion_force, 'cohesion'),
                          (separation_force, 'separation'),
                          (alignment_force, 'alignment'),
                          (wall_force, 'wall_avoidance')]:
            if np.any(force):
                # Normalize the force
                magnitude = np.linalg.norm(force)
                if magnitude > 0:
                    force = force / magnitude
                # Apply weight and add to list
                forces.append(force * self.weights[name])
        
        # Sum all forces
        if forces:
            total_force = np.sum(forces, axis=0)
            # Normalize the total force if it exceeds max_force
            magnitude = np.linalg.norm(total_force)
            if magnitude > self.max_force:
                total_force = (total_force / magnitude) * self.max_force
        else:
            total_force = np.zeros(2)
        
        # Track total distance moved
        self.total_distance += np.linalg.norm(velocity)
        self.survival_time += 1
        
        return total_force
    
    def update_fitness_metrics(self, position: np.ndarray, velocity: np.ndarray,
                             neighbors: List[Tuple[np.ndarray, np.ndarray]],
                             world_size: Tuple[float, float]):
        """Update metrics used for fitness calculation."""
        # Most metrics are updated in calculate_move
        pass
    
    def calculate_fitness(self) -> float:
        """Calculate fitness score based on survival success.
        
        Returns:
            float: Fitness score between 0 and 1
        """
        # Normalize metrics
        survival_score = min(1.0, self.survival_time / 1000.0)  # Cap at 1000 frames
        
        # Penalize close calls but don't let them dominate
        close_call_penalty = max(0.0, 1.0 - (self.close_calls * 0.1))
        
        # Reward staying in groups (safety in numbers)
        group_score = min(1.0, self.avg_group_size / 5.0)  # Cap at group size of 5
        
        # Combine scores with weights
        fitness = (0.5 * survival_score + 
                  0.3 * close_call_penalty +
                  0.2 * group_score)
        
        return max(0.0, min(1.0, fitness))  # Ensure between 0 and 1
  