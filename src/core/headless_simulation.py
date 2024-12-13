import numpy as np
from typing import List, Type, Callable
from .bot import Bot
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
    
    def update(self):
        """Update simulation state for one frame."""
        # Get current state of all bots
        all_states = [bot.get_state() for bot in self.bots]
        
        # Update each bot
        for bot in self.bots:
            # Get neighbors within perception radius
            neighbors = []
            bot_pos = bot.position
            for other_bot in self.bots:
                if other_bot != bot:
                    other_pos, other_vel = other_bot.get_state()
                    distance = np.linalg.norm(bot_pos - other_pos)
                    if distance < bot.intelligence.perception_radius:
                        neighbors.append((other_pos, other_vel))
            
            # Update bot with scenario parameters
            bot.update(neighbors, self.world_size, **self.scenario.get_scenario_params())
            
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