import numpy as np
from typing import Tuple, List
from ..intelligences.base import Intelligence

class Bot:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, intelligence: Intelligence):
        """Initialize a bot with a position, velocity and intelligence.
        
        Args:
            position: Initial position as numpy array [x, y]
            velocity: Initial velocity as numpy array [vx, vy]
            intelligence: Intelligence instance that controls this bot
        """
        self.position = position.astype(float)
        self.velocity = velocity.astype(float)
        self.acceleration = np.zeros(2)
        self.intelligence = intelligence
        
    def update(self, neighbors: List[Tuple[np.ndarray, np.ndarray]], world_size: Tuple[float, float], **kwargs):
        """Update bot's position and velocity based on its intelligence.
        
        Args:
            neighbors: List of (position, velocity) tuples of nearby bots
            world_size: Tuple of (width, height) of the world
            **kwargs: Additional scenario-specific parameters
        """
        # Calculate move using intelligence
        force = self.intelligence.calculate_move(
            self.position,
            self.velocity,
            neighbors,
            world_size,
            **kwargs
        )
        
        # Apply force and update position
        self.acceleration = force
        self.velocity += self.acceleration
        
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.intelligence.max_speed:
            self.velocity = (self.velocity / speed) * self.intelligence.max_speed
            
        self.position += self.velocity
        
        # Wrap around world boundaries
        self.position[0] %= world_size[0]
        self.position[1] %= world_size[1]
        
        # Reset acceleration
        self.acceleration = np.zeros(2)
        
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current state of the bot.
        
        Returns:
            Tuple of (position, velocity)
        """
        return (self.position.copy(), self.velocity.copy()) 