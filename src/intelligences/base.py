from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class Intelligence(ABC):
    """Base class for bot intelligence implementations."""
    
    def __init__(self, max_speed: float = 2.0, max_force: float = 0.1, perception_radius: float = 50.0):
        """Initialize intelligence with configurable parameters.
        
        Args:
            max_speed: Maximum speed a bot can move
            max_force: Maximum force that can be applied to a bot
            perception_radius: How far a bot can see other bots
        """
        self.max_speed = max_speed
        self.max_force = max_force
        self.perception_radius = perception_radius

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