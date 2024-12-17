import numpy as np
from typing import List, Tuple, Dict, Any
from .base import Intelligence

class PredatorIntelligence(Intelligence):
    """Intelligence for predator bots that chase and catch prey."""
    
    def __init__(self, max_speed: float = 2.0, max_force: float = 0.15,
                 perception_radius: float = 150.0, separation_radius: float = 50.0,
                 wall_detection_distance: float = 30.0, catch_radius: float = 20.0,
                 sight_speed_ratio: float = 75.0, base_energy: float = 150.0,
                 chase_weight: float = 0.4,
                 separation_weight: float = 0.1,
                 wall_avoidance_weight: float = 0.1):
        """Initialize predator intelligence with configurable parameters.
        
        Args:
            max_speed: Maximum speed
            max_force: Maximum steering force
            perception_radius: Base perception radius before sight/speed adjustment
            separation_radius: Distance to maintain from other predators
            wall_detection_distance: Distance to start avoiding walls
            catch_radius: Distance at which prey is considered caught
            sight_speed_ratio: Ratio between sight range and speed (evolvable)
            base_energy: Total energy available to allocate between speed and sight
            chase_weight: Weight for chasing prey
            separation_weight: Weight for avoiding other predators
            wall_avoidance_weight: Weight for avoiding walls
        """
        self.sight_speed_ratio = sight_speed_ratio
        self.base_energy = base_energy
        
        # Calculate speed and sight based on ratio
        derived_speed = self.base_energy / (1 + np.log(max(sight_speed_ratio, 0.1)))
        derived_sight = derived_speed * sight_speed_ratio
        
        # Clamp values to reasonable ranges
        max_speed = min(max(derived_speed, 0.5), 4.0)
        adjusted_perception = min(max(derived_sight, 20.0), 400.0)
        
        super().__init__(max_speed, max_force, adjusted_perception)
        self.catch_radius = catch_radius
        self.color = 'darkred'
        self.trail_color = 'red'
        
        # Store configurable parameters
        self.separation_radius = separation_radius
        self.wall_margin = wall_detection_distance
        self.weights = {
            'chase': chase_weight,
            'separation': separation_weight,
            'wall_avoidance': wall_avoidance_weight
        }
        
        # Metrics for fitness calculation
        self.prey_caught = 0
        self.total_chase_distance = 0.0
        self.time_without_target = 0
        self.num_updates = 0
        self.current_target = None
    
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
                    if distance < self.separation_radius:
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
                 perception_radius: float = 100.0, separation_radius: float = 30.0,
                 wall_detection_distance: float = 50.0, close_call_distance: float = 30.0,
                 sight_speed_ratio: float = 50.0, base_energy: float = 100.0,
                 evade_weight: float = 0.4,
                 cohesion_weight: float = 0.1,
                 separation_weight: float = 0.3,
                 alignment_weight: float = 0.2,
                 wall_avoidance_weight: float = 0.1):
        """Initialize prey intelligence with configurable parameters.
        
        Args:
            max_speed: Maximum speed
            max_force: Maximum steering force
            perception_radius: Base perception radius before sight/speed adjustment
            separation_radius: Distance to maintain from other prey
            wall_detection_distance: Distance to start avoiding walls
            close_call_distance: Distance that counts as a close call with predator
            sight_speed_ratio: Ratio between sight range and speed (evolvable)
            base_energy: Total energy available to allocate between speed and sight
            evade_weight: Weight for evading predators
            cohesion_weight: Weight for staying with other prey
            separation_weight: Weight for avoiding other prey
            alignment_weight: Weight for matching velocity with nearby prey
            wall_avoidance_weight: Weight for avoiding walls
        """
        self.sight_speed_ratio = sight_speed_ratio
        self.base_energy = base_energy
        
        # Calculate speed and sight based on ratio
        derived_speed = self.base_energy / (1 + np.log(max(sight_speed_ratio, 0.1)))
        derived_sight = derived_speed * sight_speed_ratio
        
        # Clamp values to reasonable ranges
        max_speed = min(max(derived_speed, 0.5), 3.0)
        adjusted_perception = min(max(derived_sight, 20.0), 300.0)
        
        super().__init__(max_speed, max_force, adjusted_perception)
        self.color = 'lightblue'
        self.trail_color = 'blue'
        
        # Store configurable parameters
        self.desired_separation = separation_radius
        self.wall_margin = wall_detection_distance
        self.close_call_distance = close_call_distance
        self.min_cohesion_distance = separation_radius * 0.67  # 2/3 of separation radius
        
        self.weights = {
            'evade': evade_weight,
            'cohesion': cohesion_weight,
            'separation': separation_weight,
            'alignment': alignment_weight,
            'wall_avoidance': wall_avoidance_weight
        }
        
        # Metrics for fitness calculation
        self.survival_time = 0
        self.close_calls = 0
        self.total_distance = 0.0
        self.avg_group_size = 0.0
        self.num_updates = 0
    
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
  