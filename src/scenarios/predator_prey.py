import numpy as np
from typing import List, Dict, Any, Tuple, Set
from .base import Scenario

class PredatorPreyScenario(Scenario):
    """A scenario where bots are divided into predators and prey."""
    
    def __init__(self, width: float = 800, height: float = 600, enable_wrapping: bool = False, 
                 predator_ratio: float = 0.2, catch_radius: float = 10.0,
                 max_time: int = 1000, win_threshold: float = 0.90,
                 verbose: bool = True):
        """Initialize predator vs prey scenario.
        
        Args:
            width: Width of the world
            height: Height of the world
            enable_wrapping: Whether bots wrap around world edges
            predator_ratio: Ratio of predators to total bots (0.0 to 1.0)
            catch_radius: Distance at which a predator catches prey
            max_time: Maximum simulation frames before ending
            win_threshold: Ratio of prey that must be caught for predators to win
            verbose: Whether to print debug information
        """
        super().__init__(width, height, enable_wrapping)
        self.predator_ratio = predator_ratio
        self.catch_radius = catch_radius
        self.max_time = max_time
        self.win_threshold = win_threshold
        self.verbose = verbose
        
        # State tracking
        self.predator_indices: List[int] = []  # Track which bots are predators
        self.caught_prey: Set[int] = set()  # Track indices of caught prey
        self.current_time: int = 0  # Track simulation time
    
    def initialize_bots(self, num_bots: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Initialize predator and prey positions and velocities.
        
        Args:
            num_bots: Total number of bots to initialize
            
        Returns:
            List of (position, velocity) tuples
        """
        bots = []
        num_predators = int(num_bots * self.predator_ratio)
        self.predator_indices = list(range(num_predators))
        self.caught_prey.clear()
        self.current_time = 0
        
        if self.verbose:
            print(f"\nInitializing {num_bots} bots:")
            print(f"- {num_predators} predators")
            print(f"- {num_bots - num_predators} prey")
        
        # Initialize predators in one area and prey in another for visual clarity
        for i in range(num_bots):
            if i < num_predators:
                # Predators start on the left side
                position = np.array([
                    np.random.uniform(0, self.width * 0.2),  # Left 20% of world
                    np.random.uniform(0, self.height)
                ])
            else:
                # Prey start on the right side
                position = np.array([
                    np.random.uniform(self.width * 0.8, self.width),  # Right 20% of world
                    np.random.uniform(0, self.height)
                ])
            
            velocity = np.array([
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
            ])
            bots.append((position, velocity))
        
        return bots
    
    def get_specific_params(self) -> Dict[str, Any]:
        """Get predator vs prey specific parameters.
        
        Returns:
            Dictionary of parameters specific to this scenario
        """
        return {
            "predator_ratio": self.predator_ratio,
            "predator_indices": self.predator_indices,
            "catch_radius": self.catch_radius,
            "caught_prey": list(self.caught_prey),  # Convert set to list for serialization
            "current_time": self.current_time,
            "max_time": self.max_time
        }
    
    def check_completion(self, positions: List[np.ndarray], velocities: List[np.ndarray]) -> bool:
        """Check if all prey have been caught or escaped.
        
        Args:
            positions: List of current bot positions
            velocities: List of current bot velocities
            
        Returns:
            True if scenario is complete (all prey caught/escaped), False otherwise
        """
        self.current_time += 1
        
        # Check for new catches
        for pred_idx in self.predator_indices:
            pred_pos = positions[pred_idx]
            # Check each uncaught prey
            for prey_idx in range(len(positions)):
                if (prey_idx not in self.predator_indices and  # is prey
                    prey_idx not in self.caught_prey and       # not already caught
                    np.linalg.norm(pred_pos - positions[prey_idx]) <= self.catch_radius):
                    self.caught_prey.add(prey_idx)
                    if self.verbose:
                        print(f"Predator {pred_idx} caught prey {prey_idx}!")  # Feedback for catches
        
        # Count total prey
        total_prey = len(positions) - len(self.predator_indices)
        caught_ratio = len(self.caught_prey) / total_prey if total_prey > 0 else 1.0
        
        # Print status every 100 frames
        if self.verbose and self.current_time % 100 == 0:
            print(f"Time: {self.current_time}, Caught prey: {len(self.caught_prey)}/{total_prey} ({caught_ratio*100:.1f}%)")
        
        # Check win/loss conditions
        is_complete = (caught_ratio >= self.win_threshold or  # Predators win
                      self.current_time >= self.max_time)    # Time's up
        
        if is_complete and self.verbose:
            if caught_ratio >= self.win_threshold:
                print(f"\nPredators win! Caught {len(self.caught_prey)}/{total_prey} prey ({caught_ratio*100:.1f}%)")
            else:
                print(f"\nTime's up! Prey win! Only {len(self.caught_prey)}/{total_prey} caught ({caught_ratio*100:.1f}%)")
        
        return is_complete