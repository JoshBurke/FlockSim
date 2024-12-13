import numpy as np
from typing import List, Tuple
from .base import Intelligence

class FlockingIntelligence(Intelligence):
    """Implementation of flocking behavior with cohesion, alignment, separation, and wall avoidance."""
    
    def __init__(self, 
                 cohesion_weight: float = 1.5,
                 alignment_weight: float = 3.0,
                 separation_weight: float = 0.8,
                 wall_avoidance_weight: float = 2.5,
                 wall_detection_distance: float = 50.0,
                 leader_bias: float = 4.0,
                 max_speed: float = 2.0,
                 max_force: float = 0.1,
                 perception_radius: float = 60.0,
                 separation_radius: float = 25.0):
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
        self.cohesion_weight = cohesion_weight
        self.alignment_weight = alignment_weight
        self.separation_weight = separation_weight
        self.wall_avoidance_weight = wall_avoidance_weight
        self.wall_detection_distance = wall_detection_distance
        self.leader_bias = leader_bias
        self.separation_radius = min(separation_radius, perception_radius)  # Ensure it's not larger than perception
        
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
        min_weight = 1.0 / (1.0 + self.leader_bias)
        max_weight = 2.0 - min_weight
        weight = min_weight + (max_weight - min_weight) * (alignment + 1) / 2
        
        return weight
        
    def calculate_move(self, position: np.ndarray, velocity: np.ndarray, 
                      neighbors: List[Tuple[np.ndarray, np.ndarray]], 
                      world_size: Tuple[float, float], **kwargs) -> np.ndarray:
        if not neighbors:
            # If no neighbors, just avoid walls
            wall_force = self._wall_avoidance(position, velocity, world_size) * self.wall_avoidance_weight
            return np.clip(wall_force, -self.max_force, self.max_force)
        
        # Get neighbors within separation radius (subset of all neighbors)
        close_neighbors = [
            (n_pos, n_vel) for n_pos, n_vel in neighbors 
            if np.linalg.norm(n_pos - position) < self.separation_radius
        ]
            
        # Calculate flocking forces
        cohesion = self._cohesion(position, velocity, neighbors) * self.cohesion_weight
        alignment = self._alignment(position, velocity, neighbors) * self.alignment_weight
        separation = self._separation(position, close_neighbors) * self.separation_weight
        wall_force = self._wall_avoidance(position, velocity, world_size) * self.wall_avoidance_weight
        
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