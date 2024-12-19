import numpy as np
from typing import List, Type, Callable, Dict, Tuple
from .bot import Bot
from .spatial_grid import SpatialGrid
from ..scenarios.base import Scenario
from ..intelligences.base import Intelligence

class HeadlessSimulation:
    """A non-visualized simulation for fast evaluation during learning."""
    
    def __init__(self, scenario: Scenario, 
                 intelligence_factories: Dict[str, Callable[[], Intelligence]], 
                 population_indices: Dict[str, List[int]]):
        """Initialize headless simulation.
        
        Args:
            scenario: Scenario instance
            intelligence_factories: Dict mapping population names to their intelligence factories
            population_indices: Dict mapping population names to their bot indices
        """
        self.scenario = scenario
        self.world_size = scenario.get_world_bounds()
        
        # Initialize bots with scenario-provided positions and velocities
        total_bots = sum(len(indices) for indices in population_indices.values())
        bot_states = scenario.initialize_bots(total_bots)
        self.bots = []
        
        # Create bots using appropriate factories based on indices
        for i, (position, velocity) in enumerate(bot_states):
            for pop_name, indices in population_indices.items():
                if i in indices:
                    intelligence = intelligence_factories[pop_name]()
                    self.bots.append(Bot(position, velocity, intelligence))
                    break
        
        # Create spatial grid for efficient neighbor finding
        # Use the maximum perception radius among all bots for cell size
        max_perception = max(bot.intelligence.perception_radius for bot in self.bots)
        self.spatial_grid = SpatialGrid(
            width=self.world_size[0],
            height=self.world_size[1],
            cell_size=max_perception
        )
        
        # Initialize bot positions in grid
        for i, bot in enumerate(self.bots):
            self.spatial_grid.update_object(i, bot.position)
        
        # Cache bot states to avoid repeated calls
        self.bot_states: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        
        # Store population information for fitness calculation
        self.population_indices = population_indices
    
    def update(self):
        """Update simulation state for one frame."""
        # Update cached bot states
        for i, bot in enumerate(self.bots):
            self.bot_states[i] = bot.get_state()
        
        # Update each bot
        for i, bot in enumerate(self.bots):
            # Get potential neighbors from spatial grid
            potential_neighbors = self.spatial_grid.get_potential_neighbors(bot.position)
            
            # Filter neighbors by actual distance
            neighbors = []
            neighbor_indices = []  # Track indices of neighbors
            bot_pos = bot.position
            perception_radius = bot.intelligence.perception_radius
            perception_radius_sq = perception_radius * perception_radius
            
            for n_id in potential_neighbors:
                if n_id != i:  # Skip self
                    n_pos, n_vel = self.bot_states[n_id]
                    # Use squared distance to avoid sqrt
                    offset = n_pos - bot_pos
                    dist_sq = np.dot(offset, offset)
                    if dist_sq < perception_radius_sq:
                        neighbors.append((n_pos, n_vel))
                        neighbor_indices.append(n_id)
            
            # Get scenario parameters and add bot-specific parameters
            scenario_params = self.scenario.get_scenario_params()
            scenario_params['bot_index'] = i
            scenario_params['neighbor_indices'] = neighbor_indices
            
            # Update bot with scenario parameters
            bot.update(neighbors, self.world_size, **scenario_params)
            
            # Update spatial grid with new position
            self.spatial_grid.update_object(i, bot.position)
            
            # Update fitness metrics
            bot.intelligence.update_fitness_metrics(
                bot.position,
                bot.velocity,
                neighbors,
                self.world_size
            )
    
    def run(self, frames: int = 500) -> Dict[str, List[float]]:
        """Run simulation for specified number of frames.
        
        Args:
            frames: Number of frames to simulate
            
        Returns:
            Dict mapping population names to lists of fitness scores for each member
        """
        for _ in range(frames):
            self.update()
            
            # Check scenario completion
            positions = [bot.position for bot in self.bots]
            velocities = [bot.velocity for bot in self.bots]
            if self.scenario.check_completion(positions, velocities):
                break
        
        # Calculate fitness scores for each population
        fitness_scores = {}
        for pop_name, indices in self.population_indices.items():
            fitness_scores[pop_name] = [
                self.bots[i].intelligence.calculate_fitness()
                for i in indices
            ]
        
        return fitness_scores 