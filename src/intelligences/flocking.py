import numpy as np
from typing import List, Tuple
from .base import Intelligence

class FlockingIntelligence(Intelligence):
    """Implementation of flocking behavior with cohesion, alignment, and separation."""
    
    def __init__(self, cohesion_weight=1.5, alignment_weight=1.8, separation_weight=0.8):
        super().__init__()
        self.cohesion_weight = cohesion_weight
        self.alignment_weight = alignment_weight
        self.separation_weight = separation_weight
        
    def calculate_move(self, position: np.ndarray, velocity: np.ndarray, 
                      neighbors: List[Tuple[np.ndarray, np.ndarray]], 
                      world_size: Tuple[float, float], **kwargs) -> np.ndarray:
        if not neighbors:
            return np.zeros(2)
            
        # Calculate flocking forces
        cohesion = self._cohesion(position, velocity, neighbors) * self.cohesion_weight
        alignment = self._alignment(position, velocity, neighbors) * self.alignment_weight
        separation = self._separation(position, neighbors) * self.separation_weight
        
        # Combine forces
        total_force = cohesion + alignment + separation
        return np.clip(total_force, -self.max_force, self.max_force)
        
    def _cohesion(self, position: np.ndarray, velocity: np.ndarray, 
                  neighbors: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        weighted_positions = []
        weights = []
        
        for n_pos, _ in neighbors:
            # Calculate direction to neighbor
            to_neighbor = n_pos - position
            # Calculate dot product with current velocity
            norm_velocity = velocity / np.linalg.norm(velocity) if np.linalg.norm(velocity) > 0 else np.zeros(2)
            norm_to_neighbor = to_neighbor / np.linalg.norm(to_neighbor) if np.linalg.norm(to_neighbor) > 0 else np.zeros(2)
            front_weight = np.dot(norm_velocity, norm_to_neighbor)
            # Remap from [-1, 1] to [0.5, 2.0]
            front_weight = 1.75 + front_weight * 0.75
            
            weighted_positions.append(n_pos * front_weight)
            weights.append(front_weight)
        
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
            # Calculate direction to neighbor
            to_neighbor = n_pos - position
            # Calculate dot product with current velocity
            norm_velocity = velocity / np.linalg.norm(velocity) if np.linalg.norm(velocity) > 0 else np.zeros(2)
            norm_to_neighbor = to_neighbor / np.linalg.norm(to_neighbor) if np.linalg.norm(to_neighbor) > 0 else np.zeros(2)
            front_weight = np.dot(norm_velocity, norm_to_neighbor)
            # Remap from [-1, 1] to [0.5, 2.0]
            front_weight = 1.75 + front_weight * 0.75
            
            weighted_velocities.append(n_vel * front_weight)
            weights.append(front_weight)
        
        # Calculate weighted average velocity
        avg_vel = np.average(weighted_velocities, axis=0, weights=weights)
        
        if np.linalg.norm(avg_vel) > 0:
            avg_vel = (avg_vel / np.linalg.norm(avg_vel)) * self.max_speed
            
        return avg_vel - velocity
        
    def _separation(self, position: np.ndarray, neighbors: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        steer = np.zeros(2)
        
        for n_pos, _ in neighbors:
            diff = position - n_pos
            dist = np.linalg.norm(diff)
            if dist > 0:
                # Weight by distance (closer neighbors have stronger effect)
                steer += (diff / dist) / dist
        
        if np.linalg.norm(steer) > 0:
            steer = (steer / np.linalg.norm(steer)) * self.max_speed
            
        return steer 