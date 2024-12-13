from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class Intelligence(ABC):
    """Base class for bot intelligence implementations."""
    
    def __init__(self):
        self.max_speed = 2.0
        self.max_force = 0.1
        self.perception_radius = 50

    @abstractmethod
    def calculate_move(self, 
                      position: np.ndarray,
                      velocity: np.ndarray,
                      neighbors: List[Tuple[np.ndarray, np.ndarray]],
                      world_size: Tuple[float, float],
                      **kwargs) -> np.ndarray:
        """Calculate the next move for the bot.
        
        Args:
            position: Current position as numpy array [x, y]
            velocity: Current velocity as numpy array [vx, vy]
            neighbors: List of tuples containing (position, velocity) of nearby bots
            world_size: Tuple of (width, height) of the world
            **kwargs: Additional scenario-specific parameters
            
        Returns:
            numpy.ndarray: The force to apply to the bot
        """
        pass

    def set_parameters(self, **kwargs):
        """Update intelligence parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 