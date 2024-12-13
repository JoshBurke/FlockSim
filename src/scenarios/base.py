from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np

class Scenario(ABC):
    """Base class for simulation scenarios."""
    
    def __init__(self, width: float = 800, height: float = 600, enable_wrapping: bool = False):
        """Initialize scenario with configurable parameters.
        
        Args:
            width: Width of the world
            height: Height of the world
            enable_wrapping: Whether bots wrap around world edges (False = bounce off walls)
        """
        self.width = width
        self.height = height
        self.enable_wrapping = enable_wrapping
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
    
    def get_scenario_params(self) -> Dict[str, Any]:
        """Get all scenario parameters including base and scenario-specific ones.
        
        Returns:
            Dictionary of parameters for this scenario
        """
        # Base parameters that all scenarios need
        params = {
            "enable_wrapping": self.enable_wrapping
        }
        
        # Get scenario-specific parameters
        scenario_params = self.get_specific_params()
        params.update(scenario_params)
        
        return params
    
    def get_specific_params(self) -> Dict[str, Any]:
        """Get scenario-specific parameters. Override this in subclasses.
        
        Returns:
            Dictionary of parameters specific to this scenario
        """
        return {}
    
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