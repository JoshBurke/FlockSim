from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

class Intelligence(ABC):
    """Base class for bot intelligence implementations."""
    
    def __init__(self, max_speed: float = 2.0, max_force: float = 0.1, perception_radius: float = 50.0):
        """Initialize intelligence with configurable parameters."""
        self.max_speed = max_speed
        self.max_force = max_force
        self.perception_radius = perception_radius
        self.weights: Dict[str, float] = {}  # To be populated by subclasses
        self.color = 'blue'  # Default color, should be overridden by subclasses
        self.trail_color = 'red'  # Color for velocity/direction indicator
        
    @abstractmethod
    def calculate_move(self, 
                      position: np.ndarray,
                      velocity: np.ndarray,
                      neighbors: List[Tuple[np.ndarray, np.ndarray]],
                      world_size: Tuple[float, float],
                      **kwargs) -> np.ndarray:
        """Calculate the next move for the bot."""
        pass
    
    @abstractmethod
    def update_fitness_metrics(self,
                             position: np.ndarray,
                             velocity: np.ndarray,
                             neighbors: List[Tuple[np.ndarray, np.ndarray]],
                             world_size: Tuple[float, float]):
        """Update internal fitness metrics based on current state.
        
        Each intelligence implementation should track its own relevant metrics.
        """
        pass
    
    @abstractmethod
    def calculate_fitness(self) -> float:
        """Calculate fitness score based on intelligence-specific criteria.
        
        Returns:
            float: Fitness score between 0 and 1
        """
        pass
    
    def mutate(self, mutation_rate: float = 0.1, mutation_range: float = 0.2):
        """Mutate the bot's weights randomly."""
        for key in self.weights:
            if np.random.random() < mutation_rate:
                self.weights[key] *= (1 + np.random.uniform(-mutation_range, mutation_range))
    
    def crossover(self, other: 'Intelligence') -> Dict[str, float]:
        """Create a new set of weights by crossing with another intelligence."""
        new_weights = {}
        for key in self.weights:
            # Randomly choose weight from either parent
            if np.random.random() < 0.5:
                new_weights[key] = self.weights[key]
            else:
                new_weights[key] = other.weights[key]
        return new_weights
    
    def set_weights(self, weights: Dict[str, float]):
        """Set the bot's weights."""
        self.weights.update(weights)
    
    def get_weights(self) -> Dict[str, float]:
        """Get the bot's current weights."""
        return self.weights.copy()
    
    def set_parameters(self, **kwargs):
        """Update intelligence parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)