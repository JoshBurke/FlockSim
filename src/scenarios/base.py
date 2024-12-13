from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np

class Scenario(ABC):
    """Base class for simulation scenarios."""
    
    def __init__(self, width: float = 800, height: float = 600):
        self.width = width
        self.height = height
        self.obstacles: List[Dict[str, Any]] = []
        self.objectives: List[Dict[str, Any]] = []
        
    @abstractmethod
    def initialize_bots(self, num_bots: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Initialize bot positions and velocities for this scenario.
        
        Args:
            num_bots: Number of bots to initialize
            
        Returns:
            List of tuples containing (position, velocity) for each bot
        """
        pass
    
    @abstractmethod
    def get_scenario_params(self) -> Dict[str, Any]:
        """Get scenario-specific parameters to pass to intelligence.
        
        Returns:
            Dictionary of parameters specific to this scenario
        """
        pass
    
    @abstractmethod
    def check_completion(self, positions: List[np.ndarray], velocities: List[np.ndarray]) -> bool:
        """Check if the scenario's objectives have been completed.
        
        Args:
            positions: List of current bot positions
            velocities: List of current bot velocities
            
        Returns:
            bool: True if scenario is complete, False otherwise
        """
        pass
    
    def get_world_bounds(self) -> Tuple[float, float]:
        """Get the world size.
        
        Returns:
            Tuple of (width, height)
        """
        return (self.width, self.height) 