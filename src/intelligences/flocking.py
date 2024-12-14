import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from .base import Intelligence
from ..config import FLOCKING_DEFAULTS

@dataclass
class FlockingMetrics:
    """Metrics specific to flocking behavior."""
    # Cohesion metrics
    avg_distance_to_center: float = 0.0
    flock_spread: float = 0.0
    
    # Alignment metrics
    avg_velocity_alignment: float = 0.0
    velocity_consistency: float = 0.0
    
    # Separation metrics
    min_neighbor_distance: float = float('inf')
    separation_violations: int = 0
    
    # Wall interaction
    wall_collisions: int = 0
    wall_distance: float = 0.0
    
    # Group dynamics
    flock_size: float = 0.0
    time_in_flock: int = 0
    
    # Tracking
    update_count: int = 0

class FlockingIntelligence(Intelligence):
    """Implementation of flocking behavior with cohesion, alignment, separation, and wall avoidance."""
    
    def __init__(self, 
                 cohesion_weight: float = FLOCKING_DEFAULTS['cohesion'],
                 alignment_weight: float = FLOCKING_DEFAULTS['alignment'],
                 separation_weight: float = FLOCKING_DEFAULTS['separation'],
                 wall_avoidance_weight: float = FLOCKING_DEFAULTS['wall_avoidance'],
                 wall_detection_distance: float = 50.0,
                 leader_bias: float = FLOCKING_DEFAULTS['leader_bias'],
                 max_speed: float = 2.0,
                 max_force: float = 0.1,
                 perception_radius: float = FLOCKING_DEFAULTS['perception_radius'],
                 separation_radius: float = FLOCKING_DEFAULTS['separation_radius']):
        """Initialize flocking intelligence with configurable parameters.
        
        Args:
            cohesion_weight: Weight of cohesion force
            alignment_weight: Weight of alignment force (high value for strong flock synchronization)
            separation_weight: Weight of separation force
            wall_avoidance_weight: Weight of wall avoidance force
            wall_detection_distance: Distance at which to start avoiding walls
            leader_bias: How much to favor bots in front (1.0 = equal, >1.0 = favor front)
            max_speed: Maximum speed a bot can move
            max_force: Maximum force that can be applied to a bot
            perception_radius: How far a bot can see other bots
            separation_radius: Distance at which separation force starts (should be <= perception_radius)
        """
        super().__init__(max_speed=max_speed, max_force=max_force, perception_radius=perception_radius)
        # Initialize weights dictionary for evolution
        self.weights = {
            'cohesion': cohesion_weight,
            'alignment': alignment_weight,
            'separation': separation_weight,
            'wall_avoidance': wall_avoidance_weight,
            'leader_bias': leader_bias
        }
        self.wall_detection_distance = wall_detection_distance
        self.separation_radius = min(separation_radius, perception_radius)
        # Initialize fitness tracking
        self.metrics = FlockingMetrics()
        
    def update_fitness_metrics(self,
                             position: np.ndarray,
                             velocity: np.ndarray,
                             neighbors: List[Tuple[np.ndarray, np.ndarray]],
                             world_size: Tuple[float, float]):
        """Update flocking-specific fitness metrics."""
        self.metrics.update_count += 1
        
        if neighbors:
            # Calculate center of flock
            positions = [n_pos for n_pos, _ in neighbors] + [position]
            center = np.mean(positions, axis=0)
            
            # Update cohesion metrics
            distances_to_center = [np.linalg.norm(pos - center) for pos in positions]
            self.metrics.avg_distance_to_center = np.mean(distances_to_center)
            self.metrics.flock_spread = np.std(distances_to_center)
            
            # Update alignment metrics
            velocities = [n_vel for _, n_vel in neighbors] + [velocity]
            avg_velocity = np.mean(velocities, axis=0)
            if np.linalg.norm(avg_velocity) > 0:
                alignments = [
                    np.dot(vel, avg_velocity) / (np.linalg.norm(vel) * np.linalg.norm(avg_velocity))
                    for vel in velocities if np.linalg.norm(vel) > 0
                ]
                self.metrics.avg_velocity_alignment = np.mean(alignments)
            
            # Update separation metrics
            neighbor_distances = [np.linalg.norm(n_pos - position) for n_pos, _ in neighbors]
            if neighbor_distances:
                self.metrics.min_neighbor_distance = min(neighbor_distances)
                self.metrics.separation_violations += sum(
                    1 for d in neighbor_distances if d < self.separation_radius
                )
            
            # Update group metrics
            self.metrics.flock_size = len(neighbors)
            self.metrics.time_in_flock += 1
        
        # Update wall metrics
        wall_distances = [
            position[0], world_size[0] - position[0],
            position[1], world_size[1] - position[1]
        ]
        min_wall_dist = min(wall_distances)
        self.metrics.wall_distance = min_wall_dist
        if min_wall_dist <= 0:
            self.metrics.wall_collisions += 1
    
    def calculate_fitness(self) -> float:
        """Calculate fitness score for flocking behavior."""
        if self.metrics.update_count == 0:
            return 0.0
            
        # Calculate component scores (0 to 1)
        cohesion_score = np.clip(1.0 - (self.metrics.avg_distance_to_center / self.perception_radius), 0, 1)
        alignment_score = (self.metrics.avg_velocity_alignment + 1) / 2  # Convert from [-1,1] to [0,1]
        separation_score = 1.0 - min(1.0, self.metrics.separation_violations / self.metrics.update_count)
        wall_score = 1.0 - min(1.0, self.metrics.wall_collisions / 10)  # Allow up to 10 collisions
        flock_score = self.metrics.time_in_flock / self.metrics.update_count
        
        # Weight the components
        component_weights = {
            'cohesion': 1.0,
            'alignment': 1.2,
            'separation': 1.5,
            'wall_avoidance': 2.0,
            'flocking': 1.3
        }
        
        weighted_scores = [
            cohesion_score * component_weights['cohesion'],
            alignment_score * component_weights['alignment'],
            separation_score * component_weights['separation'],
            wall_score * component_weights['wall_avoidance'],
            flock_score * component_weights['flocking']
        ]
        
        return sum(weighted_scores) / sum(component_weights.values())
        
    def _calculate_influence_weight(self, velocity: np.ndarray, to_neighbor: np.ndarray) -> float:
        """Calculate how much influence a neighbor should have based on their relative position.
        
        Args:
            velocity: Current bot's velocity
            to_neighbor: Vector pointing to the neighbor
            
        Returns:
            float: Weight factor for this neighbor's influence
        """
        if np.linalg.norm(velocity) == 0 or np.linalg.norm(to_neighbor) == 0:
            return 1.0
            
        # Calculate dot product to determine if neighbor is in front
        norm_velocity = velocity / np.linalg.norm(velocity)
        norm_to_neighbor = to_neighbor / np.linalg.norm(to_neighbor)
        alignment = np.dot(norm_velocity, norm_to_neighbor)  # -1 to 1
        
        # Transform alignment to a weight using leader_bias
        # For leader_bias = 1.0: weight range is [0.5, 1.5]
        # For leader_bias = 2.0: weight range is [0.1, 1.9]
        # For leader_bias = 4.0: weight range is [0.01, 1.99]
        min_weight = 1.0 / (1.0 + self.weights['leader_bias'])
        max_weight = 2.0 - min_weight
        weight = min_weight + (max_weight - min_weight) * (alignment + 1) / 2
        
        return weight
        
    def calculate_move(self, position: np.ndarray, velocity: np.ndarray, 
                      neighbors: List[Tuple[np.ndarray, np.ndarray]], 
                      world_size: Tuple[float, float], **kwargs) -> np.ndarray:
        if not neighbors:
            # If no neighbors, just avoid walls
            wall_force = self._wall_avoidance(position, velocity, world_size) * self.weights['wall_avoidance']
            return np.clip(wall_force, -self.max_force, self.max_force)
        
        # Get neighbors within separation radius (subset of all neighbors)
        close_neighbors = [
            (n_pos, n_vel) for n_pos, n_vel in neighbors 
            if np.linalg.norm(n_pos - position) < self.separation_radius
        ]
            
        # Calculate flocking forces
        cohesion = self._cohesion(position, velocity, neighbors) * self.weights['cohesion']
        alignment = self._alignment(position, velocity, neighbors) * self.weights['alignment']
        separation = self._separation(position, close_neighbors) * self.weights['separation']
        wall_force = self._wall_avoidance(position, velocity, world_size) * self.weights['wall_avoidance']
        
        # Combine forces
        total_force = cohesion + alignment + separation + wall_force
        return np.clip(total_force, -self.max_force, self.max_force)
        
    def _wall_avoidance(self, position: np.ndarray, velocity: np.ndarray, world_size: Tuple[float, float]) -> np.ndarray:
        """Calculate force to avoid walls."""
        force = np.zeros(2)
        
        # Check distance to each wall and add repulsive force if too close
        walls = [
            (position[0], 0),  # Distance to left wall
            (position[0], world_size[0]),  # Distance to right wall
            (position[1], 0),  # Distance to bottom wall
            (position[1], world_size[1])  # Distance to top wall
        ]
        
        for i, (pos, wall) in enumerate(walls):
            distance = abs(pos - wall)
            if distance < self.wall_detection_distance:
                # Calculate repulsion strength (stronger as we get closer)
                strength = (1 - distance/self.wall_detection_distance) * self.max_speed
                
                # Add repulsive force away from wall
                if i < 2:  # Horizontal walls
                    # If close to left wall, push right; if close to right wall, push left
                    direction = 1 if wall == 0 else -1
                    force[0] += direction * strength
                else:  # Vertical walls
                    # If close to bottom wall, push up; if close to top wall, push down
                    direction = 1 if wall == 0 else -1
                    force[1] += direction * strength
                    
                # Add some "bounce" to the force based on current velocity
                # This helps prevent getting stuck in corners
                if distance < self.wall_detection_distance * 0.5:
                    bounce = velocity * -0.5  # Reflect some of the current velocity
                    force += bounce
        
        return force
        
    def _cohesion(self, position: np.ndarray, velocity: np.ndarray, 
                  neighbors: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        weighted_positions = []
        weights = []
        
        for n_pos, _ in neighbors:
            to_neighbor = n_pos - position
            weight = self._calculate_influence_weight(velocity, to_neighbor)
            weighted_positions.append(n_pos * weight)
            weights.append(weight)
        
        # Calculate weighted center
        center = np.average(weighted_positions, axis=0, weights=weights)
        
        # Create steering force towards center
        desired = center - position
        if np.linalg.norm(desired) > 0:
            desired = (desired / np.linalg.norm(desired)) * self.max_speed
        
        return desired - velocity
        
    def _alignment(self, position: np.ndarray, velocity: np.ndarray, 
                  neighbors: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        weighted_velocities = []
        weights = []
        
        for n_pos, n_vel in neighbors:
            to_neighbor = n_pos - position
            weight = self._calculate_influence_weight(velocity, to_neighbor)
            weighted_velocities.append(n_vel * weight)
            weights.append(weight)
        
        # Calculate weighted average velocity
        avg_vel = np.average(weighted_velocities, axis=0, weights=weights)
        
        if np.linalg.norm(avg_vel) > 0:
            avg_vel = (avg_vel / np.linalg.norm(avg_vel)) * self.max_speed
            
        return avg_vel - velocity
        
    def _separation(self, position: np.ndarray, neighbors: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Calculate separation force using inverse square law for smoother transitions."""
        if not neighbors:
            return np.zeros(2)
            
        steer = np.zeros(2)
        
        for n_pos, _ in neighbors:
            diff = position - n_pos
            dist = np.linalg.norm(diff)
            if dist > 0:
                # Use inverse square law for smoother force falloff
                # Normalize by separation radius to make force consistent across different radii
                normalized_dist = dist / self.separation_radius
                repulsion = 1.0 / (normalized_dist * normalized_dist)
                steer += (diff / dist) * repulsion
        
        if np.linalg.norm(steer) > 0:
            steer = (steer / np.linalg.norm(steer)) * self.max_speed
            
        return steer 