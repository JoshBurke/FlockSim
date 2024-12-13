import numpy as np
from typing import List, Type, Callable, Dict, Tuple
from .bot import Bot
from .spatial_grid import SpatialGrid
from ..scenarios.base import Scenario
from ..intelligences.base import Intelligence

class HeadlessSimulation:
    """A non-visualized simulation for fast evaluation during learning."""
    
    def __init__(self, scenario: Scenario, intelligence_factory: Callable[[], Intelligence], num_bots: int = 30):
        """Initialize headless simulation.
        
        Args:
            scenario: Scenario instance defining the simulation environment
            intelligence_factory: Function that creates new intelligence instances
            num_bots: Number of bots to simulate
        """
        self.scenario = scenario
        self.world_size = scenario.get_world_bounds()
        
        # Initialize bots with scenario-provided positions and velocities
        bot_states = scenario.initialize_bots(num_bots)
        self.bots = [
            Bot(position, velocity, intelligence_factory())
            for position, velocity in bot_states
        ]
        
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
            bot_pos = bot.position
            perception_radius = bot.intelligence.perception_radius
            perception_radius_sq = perception_radius * perception_radius  # Square once, compare many
            
            for n_id in potential_neighbors:
                if n_id != i:  # Skip self
                    n_pos, n_vel = self.bot_states[n_id]
                    # Use squared distance to avoid sqrt
                    offset = n_pos - bot_pos
                    dist_sq = np.dot(offset, offset)
                    if dist_sq < perception_radius_sq:
                        neighbors.append((n_pos, n_vel))
            
            # Update bot with scenario parameters
            bot.update(neighbors, self.world_size, **self.scenario.get_scenario_params())
            
            # Update spatial grid with new position
            self.spatial_grid.update_object(i, bot.position)
            
            # Update fitness metrics
            bot.intelligence.update_fitness_metrics(
                bot.position,
                bot.velocity,
                neighbors,
                self.world_size
            )
    
    def run(self, frames: int = 500) -> List[float]:
        """Run simulation for specified number of frames.
        
        Args:
            frames: Number of frames to simulate
            
        Returns:
            List of fitness scores for each bot
        """
        for _ in range(frames):
            self.update()
            
            # Check scenario completion
            positions = [bot.position for bot in self.bots]
            velocities = [bot.velocity for bot in self.bots]
            if self.scenario.check_completion(positions, velocities):
                break
        
        # Return fitness scores
        return [bot.intelligence.calculate_fitness() for bot in self.bots] 